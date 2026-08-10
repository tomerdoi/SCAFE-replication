from src.datasets.paper_benchmarks import BenchmarkRecord, _normal_name


def test_normal_name_matches_repository_filename_variants():
    assert _normal_name("Virgo_ZodiacToken") == "virgozodiactoken"
    assert _normal_name("VirgoZodiacToken.sol") == "virgozodiactokensol"


def test_benchmark_record_is_source_and_label_oriented():
    record = BenchmarkRecord(
        benchmark="TRACE-CVE",
        record_id="CVE-2021-34272",
        source_path="/tmp/CVE-2021-34272.sol",
        label_status="vulnerable",
        vuln_type="access_control",
    )
    assert record.benchmark == "TRACE-CVE"
    assert record.label_status == "vulnerable"
