from cli_test_framework.runners.json_runner import JSONRunner
from cli_test_framework.utils.report_generator import ReportGenerator
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    runner = JSONRunner(config_file=str(script_dir / "test_cases.json"), workspace=str(script_dir))
    success = runner.run_tests()
    
    # Generate and save the report
    report_generator = ReportGenerator(runner.results, str(script_dir / "test_report.txt"))
    report_generator.print_report()
    report_generator.save_report()

    runner1=JSONRunner(config_file=str(script_dir / "benchmark"/"test_cases.json"), workspace=str(script_dir))

    success1 = runner1.run_tests()
    report_generator1 = ReportGenerator(runner1.results, str(script_dir / "test_benchmark_report.txt"))
    report_generator1.print_report()
    report_generator1.save_report()

    sys.exit(0 if success and success1 else 1)

if __name__ == "__main__":
    main()