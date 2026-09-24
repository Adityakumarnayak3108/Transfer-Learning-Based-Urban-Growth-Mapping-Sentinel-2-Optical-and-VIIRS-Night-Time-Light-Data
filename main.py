import subprocess

steps = [
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/visualize.py"
]

for step in steps:
    print(f"\n{'='*60}")
    print(f"Running {step}")
    print(f"{'='*60}\n")

    subprocess.run(["python", step], check=True)

print("\nProject Completed Successfully!")