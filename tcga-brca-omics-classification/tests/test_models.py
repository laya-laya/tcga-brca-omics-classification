from tcga_brca.models import search_spaces

def test_search_spaces_and_pipeline_order():
    spaces = search_spaces()
    assert set(spaces) == {"elastic_net", "random_forest"}

    for _, (pipeline, _) in spaces.items():
        steps = list(pipeline.named_steps)
        assert steps[0] == "select"
        assert steps[-1] == "model"
