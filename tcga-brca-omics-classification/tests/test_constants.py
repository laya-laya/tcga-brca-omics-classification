from tcga_brca.constants import PAM50_GENES
def test_pam50_count():
    assert len(PAM50_GENES) == 50
    assert "ESR1" in PAM50_GENES
    assert "ERBB2" in PAM50_GENES
