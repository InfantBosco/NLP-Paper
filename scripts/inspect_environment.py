#!/usr/bin/env python
import json
from sbert_reproduction.environment import get_environment_info

def main():
    info = get_environment_info()
    print("=== Environment & Hardware Audit ===")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
