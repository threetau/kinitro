curl -X POST "http://localhost:8080/competitions" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mt1 baby",
    "benchmarks": [
        {
            "provider": "metaworld",
            "benchmark_name": "MT1",
            "config": {
                "env_name": "reach-v3"
            }
        }
    ],
    "points": 50
  }'
