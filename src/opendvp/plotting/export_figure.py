

def export_figure(fig, path, suffix, dpi=600):
    # Ensure path ends with a slash or is joined correctly

    os.makedirs(path, exist_ok=True)

    datetime_str = time.strftime("%Y%m%d_%H%M")
    base_filename = f"{datetime_str}_{suffix}"

    pdf_path = os.path.join(path, f"{base_filename}.pdf")
    svg_path = os.path.join(path, f"{base_filename}.svg")

    fig.savefig(fname=pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(fname=svg_path, format="svg", dpi=dpi, bbox_inches="tight")

    print(f"Figure saved as: {pdf_path} and {svg_path}")