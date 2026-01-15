"""
Scoring engine for Kinitro evaluations.

Handles all scoring, eligibility checking, and leader candidate logic.
Extracted from BackendService for better separation of concerns.

The ScoringEngine uses pluggable ScoringStrategy implementations to support
different task types with their own eligibility and scoring logic.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fiber.chain.models import Node
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.log import get_logger
from core.tasks import TaskType

from .models import (
    BackendEvaluationResult,
    Competition,
    CompetitionLeaderCandidate,
    LeaderCandidateStatus,
    SS58Address,
)
from .scoring import ScoringStrategyRegistry

if TYPE_CHECKING:
    from .scoring import ScoringStrategy

logger = get_logger(__name__)


class ScoringConfig:
    """Configuration for the scoring engine."""

    def __init__(
        self,
        owner_uid: int = 4,
        burn_pct: float = 0.98,
    ):
        self.owner_uid = owner_uid
        self.burn_pct = self._validate_burn_pct(burn_pct)

    @staticmethod
    def _validate_burn_pct(burn_pct: float) -> float:
        """Validate and clamp burn percentage to [0, 1]."""
        if burn_pct < 0 or burn_pct > 1:
            logger.warning(
                "Configured burn_pct %.3f out of bounds [0, 1]; clamping.",
                burn_pct,
            )
            return max(0.0, min(1.0, burn_pct))
        return burn_pct


class ScoringEngine:
    """
    Handles all scoring, eligibility, and leader candidate logic.

    This class is responsible for:
    - Checking if miners meet eligibility criteria for competitions
    - Creating and managing leader candidates
    - Computing scores for competition winners
    - Calculating weight distributions for miners
    
    The ScoringEngine uses pluggable ScoringStrategy implementations to support
    different task types. The strategy is selected based on competition.task_type.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        config: ScoringConfig,
        id_generator,
        strategy_registry: ScoringStrategyRegistry | None = None,
    ):
        self.session_factory = session_factory
        self.config = config
        self.id_generator = id_generator
        # Use provided registry or create default with all built-in strategies
        self.strategy_registry = strategy_registry or ScoringStrategyRegistry.default()

    def get_strategy(self, competition: Competition) -> "ScoringStrategy":
        """Get the scoring strategy for a competition based on its task type.
        
        Args:
            competition: The competition to get a strategy for
            
        Returns:
            The appropriate ScoringStrategy for the competition's task type
        """
        return self.strategy_registry.get(competition.task_type)

    def is_eligible(
        self,
        result: BackendEvaluationResult,
        competition: Competition,
    ) -> bool:
        """Check if a miner meets eligibility criteria for a competition.
        
        Uses the competition's task type to select the appropriate scoring
        strategy for eligibility checking.
        """
        strategy = self.get_strategy(competition)
        metrics = strategy.extract_metrics(result)
        eligibility = strategy.check_eligibility(metrics, competition)
        
        if not eligibility.eligible and eligibility.reason:
            logger.debug(
                "Miner %s excluded from competition %s: %s",
                result.miner_hotkey,
                competition.id,
                eligibility.reason,
            )
        
        return eligibility.eligible

    async def queue_leader_candidate(
        self,
        session: AsyncSession,
        competition: Competition,
        result: BackendEvaluationResult,
    ) -> bool:
        """Persist a leader candidate if not already recorded for this result."""
        if result.avg_reward is None:
            logger.debug(
                "Skipping leader candidate creation without avg_reward: competition=%s result_id=%s",
                competition.id,
                result.id,
            )
            return False

        existing_candidate_result = await session.execute(
            select(CompetitionLeaderCandidate).where(
                CompetitionLeaderCandidate.evaluation_result_id == result.id
            )
        )
        existing_candidate = existing_candidate_result.scalar_one_or_none()
        if existing_candidate:
            logger.debug(
                "Leader candidate already exists for evaluation result %s (competition=%s)",
                result.id,
                competition.id,
            )
            return False

        candidate = CompetitionLeaderCandidate(
            id=next(self.id_generator),
            competition_id=competition.id,
            miner_hotkey=result.miner_hotkey,
            evaluation_result_id=result.id,
            avg_reward=result.avg_reward,
            success_rate=result.success_rate,
            score=result.score,
            total_episodes=result.total_episodes,
        )
        session.add(candidate)
        return True

    async def score_evaluations(self) -> dict[SS58Address, float]:
        """
        Score completed evaluations with winner-takes-all per competition.

        Scoring logic:
        - Miners must meet minimum success rate threshold per competition
        - Miners must pass minimum avg reward threshold per competition
        - Eligible challengers above the approved leader's success rate are queued for admin review
        - If the current leader improves or matches their approved success rate, that result is also queued
        - Current leader retains position until admin approval
        - Each miner can only win ONE competition (first-win policy)
        - Final scores are normalized based on competition points

        Returns:
            dict[SS58Address, float]: Mapping of miner hotkeys to their normalized scores (0-1).
        """
        async with self.session_factory() as session:
            # Fetch all active competitions
            competitions_result = await session.execute(
                select(Competition).where(Competition.active)
            )
            competitions = competitions_result.scalars().all()

            if not competitions:
                logger.info("No active competitions found for scoring")
                return {}

            # Calculate total points across all competitions
            total_points = sum(comp.points for comp in competitions)

            # Dictionary to store winner scores
            miner_scores: dict[SS58Address, float] = {}

            for competition in competitions:
                await self._score_competition(
                    session, competition, total_points, miner_scores
                )

            # Commit any leader updates to database
            await session.commit()

            # Log final scores
            if miner_scores:
                logger.info(f"Final miner scores: {len(miner_scores)} miners scored")
                for hotkey, score in sorted(
                    miner_scores.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    logger.info(f"  {hotkey}: {score:.4f}")
            else:
                logger.info("No miners received scores")

            return miner_scores

    async def _score_competition(
        self,
        session: AsyncSession,
        competition: Competition,
        total_points: int,
        miner_scores: dict[SS58Address, float],
    ) -> None:
        """Score a single competition and update miner_scores in place."""
        # Get the scoring strategy for this competition's task type
        strategy = self.get_strategy(competition)
        
        # Get all evaluation results for this competition
        results_query = select(BackendEvaluationResult).where(
            BackendEvaluationResult.competition_id == competition.id
        )
        results = await session.execute(results_query)
        eval_results = results.scalars().all()

        if not eval_results:
            logger.debug(f"No evaluation results for competition {competition.id}")
            return

        # Find eligible challengers and order them using the strategy's compare method
        eligible_results = [
            result for result in eval_results if self.is_eligible(result, competition)
        ]

        # Sort using strategy's compare method (descending order, best first)
        from functools import cmp_to_key
        
        def compare_results(a: BackendEvaluationResult, b: BackendEvaluationResult) -> int:
            metrics_a = strategy.extract_metrics(a)
            metrics_b = strategy.extract_metrics(b)
            # Negate because we want descending order (best first)
            return -strategy.compare(metrics_a, metrics_b)
        
        eligible_results.sort(key=cmp_to_key(compare_results))

        if not eligible_results:
            if competition.current_leader_hotkey:
                logger.info(
                    "Competition %s: Current leader %s retains position (no eligible challengers)",
                    competition.id,
                    competition.current_leader_hotkey,
                )
            else:
                logger.info("Competition %s: No eligible miners found", competition.id)
            return

        current_leader = competition.current_leader_hotkey

        if current_leader is None:
            await self._handle_no_leader(session, competition, eligible_results)
        else:
            await self._handle_existing_leader(
                session, competition, current_leader, eligible_results
            )

        # Award points only to the currently approved leader
        award_hotkey = competition.current_leader_hotkey
        if not award_hotkey:
            logger.debug(
                "Competition %s: Skipping score award (no approved leader)",
                competition.id,
            )
            return

        base_score = competition.points / total_points if total_points else 0
        if base_score == 0:
            logger.debug(
                "Competition %s: Skipping zero-point competition in scoring",
                competition.id,
            )
            return

        if award_hotkey in miner_scores:
            logger.warning(
                "Miner %s already won competition - skipping score from %s. Previous score: %.4f, would have added: %.4f",
                award_hotkey,
                competition.id,
                miner_scores[award_hotkey],
                base_score * (1 - self.config.burn_pct),
            )
            return

        awarded_score = base_score * (1 - self.config.burn_pct)
        burned_score = base_score - awarded_score

        if awarded_score <= 0:
            logger.info(
                "Competition %s: Burned entire %.4f normalized score for %s (burn_pct=%.2f%%)",
                competition.id,
                base_score,
                award_hotkey,
                self.config.burn_pct * 100,
            )
            return

        miner_scores[award_hotkey] = awarded_score
        if burned_score > 0:
            logger.info(
                "Competition %s: Awarded %.4f normalized score to %s (burned %.4f; burn_pct=%.2f%%)",
                competition.id,
                awarded_score,
                award_hotkey,
                burned_score,
                self.config.burn_pct * 100,
            )
        else:
            logger.info(
                "Competition %s: Awarded %.4f normalized score to %s",
                competition.id,
                awarded_score,
                award_hotkey,
            )

    async def _handle_no_leader(
        self,
        session: AsyncSession,
        competition: Competition,
        eligible_results: list[BackendEvaluationResult],
    ) -> None:
        """Handle scoring when there is no current leader."""
        queued_any = False
        for res in eligible_results:
            created_candidate = await self.queue_leader_candidate(
                session, competition, res
            )
            if created_candidate:
                queued_any = True
                logger.info(
                    "Competition %s: Queued leader candidate %s (success_rate=%.3f, avg_reward=%.3f)",
                    competition.id,
                    res.miner_hotkey,
                    res.success_rate or 0.0,
                    res.avg_reward or 0.0,
                )
        if not queued_any:
            logger.debug(
                "Competition %s: All eligible results already queued as candidates",
                competition.id,
            )

    async def _handle_existing_leader(
        self,
        session: AsyncSession,
        competition: Competition,
        current_leader: SS58Address,
        eligible_results: list[BackendEvaluationResult],
    ) -> None:
        """Handle scoring when there is an existing leader."""
        leader_success_rate_stmt = (
            select(
                CompetitionLeaderCandidate.success_rate,
                CompetitionLeaderCandidate.evaluation_result_id,
            )
            .where(
                CompetitionLeaderCandidate.competition_id == competition.id,
                CompetitionLeaderCandidate.miner_hotkey == current_leader,
                CompetitionLeaderCandidate.status == LeaderCandidateStatus.APPROVED,
            )
            .order_by(
                CompetitionLeaderCandidate.reviewed_at.desc(),
                CompetitionLeaderCandidate.updated_at.desc(),
            )
            .limit(1)
        )
        leader_success_rate_result = await session.execute(leader_success_rate_stmt)
        leader_success_rate_row = leader_success_rate_result.first()
        leader_success_rate = (
            leader_success_rate_row[0] if leader_success_rate_row else None
        )
        leader_success_eval_id = (
            leader_success_rate_row[1] if leader_success_rate_row else None
        )
        baseline_leader_success_rate = (
            leader_success_rate if leader_success_rate is not None else -1.0
        )

        leader_best = next(
            (res for res in eligible_results if res.miner_hotkey == current_leader),
            None,
        )
        if (
            leader_best
            and leader_best.avg_reward is not None
            and leader_best.avg_reward != competition.current_leader_reward
        ):
            competition.current_leader_reward = leader_best.avg_reward
            competition.leader_updated_at = datetime.now(timezone.utc)
            logger.info(
                "Competition %s: Updated leader %s reward to %.3f",
                competition.id,
                current_leader,
                leader_best.avg_reward,
            )

        challengers: list[BackendEvaluationResult] = []
        for res in eligible_results:
            if (
                res.success_rate is not None
                and res.success_rate > baseline_leader_success_rate
            ):
                challengers.append(res)

        if (
            leader_best
            and leader_best.success_rate is not None
            and leader_best.success_rate >= baseline_leader_success_rate
            and leader_best.id != leader_success_eval_id
        ):
            challengers.append(leader_best)

        # Deduplicate challengers by evaluation_result_id while preserving order
        seen_eval_ids: set[int] = set()
        unique_challengers: list[BackendEvaluationResult] = []
        for res in challengers:
            if res.id in seen_eval_ids:
                continue
            seen_eval_ids.add(res.id)
            unique_challengers.append(res)

        for challenger in unique_challengers:
            created_candidate = await self.queue_leader_candidate(
                session, competition, challenger
            )
            if created_candidate:
                logger.info(
                    "Competition %s: Challenger %s queued for admin review (success_rate=%.3f, avg_reward=%.3f, current leader=%s success_rate=%s)",
                    competition.id,
                    challenger.miner_hotkey,
                    challenger.success_rate or 0.0,
                    challenger.avg_reward or 0.0,
                    current_leader,
                    f"{leader_success_rate:.3f}"
                    if leader_success_rate is not None
                    else "unknown",
                )
            else:
                logger.debug(
                    "Competition %s: Challenger %s already recorded as candidate",
                    competition.id,
                    challenger.miner_hotkey,
                )

    def compute_weights(
        self,
        miner_scores: dict[SS58Address, float],
        nodes: dict[SS58Address, Node],
    ) -> dict[int, float]:
        """
        Compute weight distribution from miner scores.

        Args:
            miner_scores: Mapping of miner hotkeys to their normalized scores
            nodes: Mapping of hotkeys to Node objects with node_id

        Returns:
            dict[int, float]: Mapping of UIDs to weights
        """
        weights_dict: dict[int, float] = {}

        for hotkey, weight in miner_scores.items():
            node = nodes.get(hotkey)
            if node:
                weights_dict[node.node_id] = weight

        total_weight = sum(weights_dict.values())
        if total_weight > 1.0:
            logger.warning(
                "Total miner weight %.6f exceeds 1.0 before owner allocation",
                total_weight,
            )

        owner_weight = max(0.0, 1.0 - total_weight)
        if owner_weight > 0:
            weights_dict[self.config.owner_uid] = (
                weights_dict.get(self.config.owner_uid, 0.0) + owner_weight
            )
            if all(node.node_id != self.config.owner_uid for node in nodes.values()):
                logger.warning(
                    "Owner UID %s not found in node list; assigning %.4f weight without hotkey mapping",
                    self.config.owner_uid,
                    owner_weight,
                )
            else:
                logger.info(
                    "Owner UID %s assigned remaining normalized score %.4f (burn_pct=%.2f%%)",
                    self.config.owner_uid,
                    owner_weight,
                    self.config.burn_pct * 100,
                )

        # Populate missing entries with 0.0 weight for all nodes
        for node in nodes.values():
            weights_dict.setdefault(node.node_id, 0.0)

        return weights_dict
