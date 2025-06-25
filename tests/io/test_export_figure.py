import matplotlib.pyplot as plt

from opendvp.io import export_figure

def test_export_figure_callable(tmp_path) -> None:

    fig,ax = plt.subplots()
    ax.plot([0,1],[0,2])
    
    export_figure(
        fig=fig,
        path_to_dir=tmp_path,
        suffix="test")
    
    # Get list of output files
    output_files = list(tmp_path.iterdir())
    assert len(output_files) == 2
    # Check that one file is pdf and the other is svg
    extensions = {f.suffix for f in output_files}
    assert ".pdf" in extensions
    assert ".svg" in extensions