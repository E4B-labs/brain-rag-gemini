import asyncio
import json
from pathlib import Path


async def main() -> None:
    path = Path("eval/golden.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {"cases": len(cases), "note": "Run with Vertex judge in the deployed environment."}
        )
    )
    for case in cases:
        print(
            case["id"],
            "recall@k=0.0",
            "faithfulness=0.0",
            "expected=wire service integration required",
        )


if __name__ == "__main__":
    asyncio.run(main())
