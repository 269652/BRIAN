"""Tests for README template renderer v2 with claims and citations."""
import pytest
from pathlib import Path
from neuroslm.readme_renderer_v2 import (
    TemplateRenderer,
    Claim,
    MissingMetricError,
    MissingClaimError,
    LogNotFoundError,
    _parse_claim_body,
    _claim_from_dict,
    load_metrics,
)


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repo structure."""
    root = tmp_path / "repo"
    root.mkdir()
    
    # Create metrics file
    metrics = root / "docs" / "readme_metrics.toml"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("""
[layer_a]
LAYER_A_TEST_COUNT = 42

[layer_b]
LAYER_B_BEST_GAP_RATIO = 6.55
LAYER_B_IMPROVEMENT_PCT = 64
""")
    
    # Create a log file
    logs = root / "logs" / "20260615" / "arch"
    logs.mkdir(parents=True)
    log_file = logs / "test_0_1000.log"
    log_file.write_text("""Line 1: boot
Line 2: training
Line 3: final OOD ppl=155.0
Line 4: done
""")
    
    return root


class TestClaimParsing:
    """Test claim body parsing."""
    
    def test_simple_claim(self):
        body = '''
            id: "H22",
            hypothesis: "test",
            train_ppl: 23.6
        '''
        data = _parse_claim_body(body)
        assert data['id'] == 'H22'
        assert data['hypothesis'] == 'test'
        assert data['train_ppl'] == 23.6
    
    def test_claim_with_arrays(self):
        body = '''
            id: "H1",
            back: [["file.log", 10, 20]],
            falsify: []
        '''
        data = _parse_claim_body(body)
        assert data['id'] == 'H1'
        assert data['back'] == [["file.log", 10, 20]]
        assert data['falsify'] == []
    
    def test_claim_to_object(self):
        data = {
            'id': 'H22',
            'hypothesis': 'H22',
            'train_ppl': 23.6,
            'ood_ppl': 155.0,
            'custom_field': 'value'
        }
        claim = _claim_from_dict(data)
        assert claim.id == 'H22'
        assert claim.train_ppl == 23.6
        assert claim.ood_ppl == 155.0
        assert claim['custom_field'] == 'value'


class TestTemplateRenderer:
    """Test full template rendering."""
    
    def test_metric_substitution(self, temp_repo):
        metrics = load_metrics(temp_repo / "docs" / "readme_metrics.toml")
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = "Tests: ${LAYER_A_TEST_COUNT}"
        result = renderer.render(template)
        assert result == "Tests: 42"
    
    def test_missing_metric_error(self, temp_repo):
        metrics = load_metrics(temp_repo / "docs" / "readme_metrics.toml")
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = "Value: ${MISSING_METRIC}"
        with pytest.raises(MissingMetricError) as exc_info:
            renderer.render(template)
        assert 'MISSING_METRIC' in exc_info.value.missing
    
    def test_claim_definition_and_reference(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "test_claim",
    hypothesis: "H1",
    train_ppl: 23.6,
    ood_ppl: 155.0
}

PPL: ${claim.test_claim.ood_ppl}
'''
        result = renderer.render(template)
        assert "PPL: 155.0" in result
        assert "$claim" not in result  # Definition removed
    
    def test_claim_with_checkpoint(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "H22",
    hypothesis: "H22",
    checkpoint: "hf://model/checkpoint.pt",
    train_ppl: 23.6
}

Checkpoint: ${claim.H22.checkpoint}
Train PPL: ${claim.H22.train_ppl}
'''
        result = renderer.render(template)
        assert "Checkpoint: hf://model/checkpoint.pt" in result
        assert "Train PPL: 23.6" in result
    
    def test_citation(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = "$cite(logs/20260615/arch/test_0_1000.log, 2, 3)"
        result = renderer.render(template)
        
        assert "```" in result
        assert "Line 2: training" in result
        assert "Line 3: final OOD ppl=155.0" in result
        assert "Line 4" not in result  # End is inclusive
    
    def test_citation_missing_file(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = "$cite(missing.log, 1, 5)"
        with pytest.raises(LogNotFoundError):
            renderer.render(template)
    
    def test_claim_array_access(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "H1",
    hypothesis: "H1",
    back: [["logs/20260615/arch/test_0_1000.log", 2, 3], ["other.log", 1, 2]]
}

First evidence: ${claim.H1.back[0]}
'''
        result = renderer.render(template)
        
        # Array access should expand to citation
        assert "Line 2: training" in result
        assert "Line 3: final OOD ppl=155.0" in result
    
    def test_missing_claim_reference(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = "${claim.NONEXISTENT.value}"
        with pytest.raises(MissingClaimError):
            renderer.render(template)
    
    def test_none_value_formatting(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "test",
    hypothesis: "H1",
    checkpoint: null
}

Checkpoint: ${claim.test.checkpoint}
'''
        result = renderer.render(template)
        assert "Checkpoint: —" in result
    
    def test_full_workflow(self, temp_repo):
        """Test complete rendering with all features."""
        metrics = load_metrics(temp_repo / "docs" / "readme_metrics.toml")
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "H22",
    hypothesis: "H22",
    checkpoint: "hf://checkpoint.pt",
    train_ppl: 23.6,
    ood_ppl: 155.0,
    gap_ratio: 6.55,
    back: [["logs/20260615/arch/test_0_1000.log", 3, 3]]
}

# Results

Tests passing: ${LAYER_A_TEST_COUNT}

Best run: ${claim.H22.ood_ppl} OOD PPL (gap ratio ${claim.H22.gap_ratio})

Evidence:
${claim.H22.back[0]}

Improvement: ${LAYER_B_IMPROVEMENT_PCT}%
'''
        result = renderer.render(template)
        
        # Check all substitutions worked
        assert "Tests passing: 42" in result
        assert "Best run: 155.0 OOD PPL (gap ratio 6.55)" in result
        assert "Line 3: final OOD ppl=155.0" in result
        assert "Improvement: 64%" in result
        assert "$claim" not in result
        assert "${" not in result  # All placeholders resolved


class TestMetricsLoading:
    """Test TOML metrics loading."""
    
    def test_load_flat_metrics(self, temp_repo):
        metrics = load_metrics(temp_repo / "docs" / "readme_metrics.toml")
        assert metrics['LAYER_A_TEST_COUNT'] == '42'
        assert metrics['LAYER_B_BEST_GAP_RATIO'] == '6.55'
    
    def test_numeric_to_string_conversion(self, temp_repo):
        metrics = load_metrics(temp_repo / "docs" / "readme_metrics.toml")
        # All values should be strings
        assert isinstance(metrics['LAYER_A_TEST_COUNT'], str)
        assert isinstance(metrics['LAYER_B_BEST_GAP_RATIO'], str)


class TestClaimObject:
    """Test Claim dataclass."""
    
    def test_dict_access(self):
        claim = Claim(
            id="H1",
            hypothesis="H1",
            train_ppl=23.6,
            metadata={'custom': 'value'}
        )
        assert claim['train_ppl'] == 23.6
        assert claim['custom'] == 'value'
        assert claim['hypothesis'] == 'H1'
    
    def test_optional_fields(self):
        claim = Claim(id="H1", hypothesis="H1")
        assert claim.checkpoint is None
        assert claim.train_ppl is None
        assert claim.back == []
        assert claim.falsify == []


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_template(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        assert renderer.render("") == ""
    
    def test_no_placeholders(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        template = "Plain text with no variables"
        assert renderer.render(template) == template
    
    def test_dollar_in_prose(self, temp_repo):
        """Lowercase $var should be left alone."""
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        template = "Costs $1.50/hr and $variable stays"
        assert renderer.render(template) == template
    
    def test_multiple_claims_same_hypothesis(self, temp_repo):
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        template = '''
$claim{
    id: "H22_run1",
    hypothesis: "H22",
    ood_ppl: 155.0
}

$claim{
    id: "H22_run2",
    hypothesis: "H22",
    ood_ppl: 142.1
}

Run 1: ${claim.H22_run1.ood_ppl}
Run 2: ${claim.H22_run2.ood_ppl}
'''
        result = renderer.render(template)
        assert "Run 1: 155.0" in result
        assert "Run 2: 142.1" in result
    
    def test_citation_line_ranges(self, temp_repo):
        """Test various line range edge cases."""
        metrics = {}
        renderer = TemplateRenderer(temp_repo, metrics)
        
        # Single line (start == end)
        template = "$cite(logs/20260615/arch/test_0_1000.log, 3, 3)"
        result = renderer.render(template)
        assert "Line 3: final OOD ppl=155.0" in result
        assert "Line 2" not in result
        assert "Line 4" not in result
