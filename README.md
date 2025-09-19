# Kinitro: Advancing Embodied Intelligence With Directed Incentives

Kinitro drives the future of robotic policy and planning models with incentivized competitions.

> [!NOTE]
> We'll be onboarding miners very soon. Please take a look at this repository and the [Kinitro agent template](https://github.com/threetau/kinitro-agent-template) to get an idea of how things work, and to start creating your miners. We are not running any validator code right now, so miners will not be given tasks.

## How it works

1. **Define**: Competitions are posted on the Kinitro platform, each with their own set of tasks
2. **Compete**: Miners train and submit agents.
3. **Validate & reward**: Validators evaluate the agents, and the best miners earn rewards.

For an overview on the overall architecture, please see the [architecture documentation](docs/architecture/introduction.md).

## Installation

Below are the basic installation steps for miners and validators.

1. **Clone the repository**:

    ```bash
    https://github.com/threetau/kinitro
    cd kinitro
    ```

2. **Set up environment and dependencies**:

    We need the following build dependencies:

    ```bash
    sudo apt install libpq-dev python3-dev
    ```

    Set up your Python environment:

    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv sync --dev
    uv pip install -e .
    ```

### Miner Setup
To set up a miner, please refer to the [miner documentation](docs/miner.md) for detailed instructions.

### Validator Setup
To set up a validator, please refer to the [validator documentation](docs/validator.md)

## Contributing

We welcome contributions to enhance Kinitro. Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For questions or support, please open an issue in this repository or contact the maintainers on the Kinitro or Bittensor Discord server.
