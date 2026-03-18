"""
Convierte los 5 notebooks .ipynb a HTML con sidebar nav y tema Raona.
Reemplaza "Adriana Caballero" por "G03".
"""
import re
import json
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
import plotly.io as pio

BASE = Path(__file__).parent
NB_DIR = BASE / "notebooks"
REPORT_DIR = BASE / "report"

NOTEBOOKS = [
    {
        "file": "01_data_loading.ipynb",
        "html": "nb01_data_loading.html",
        "label": "NB01: Data Loading",
        "title": "01 - Data Loading &amp; Cleaning",
        "subtitle": "Carga de datos, limpieza y preparaci&oacute;n del dataset",
    },
    {
        "file": "02_eda.ipynb",
        "html": "nb02_eda.html",
        "label": "NB02: EDA",
        "title": "02 - An&aacute;lisis Exploratorio",
        "subtitle": "5,987 contactos: distribuciones, funnel y perfilado",
    },
    {
        "file": "03_feature_engineering.ipynb",
        "html": "nb03_feature_engineering.html",
        "label": "NB03: Features",
        "title": "03 - Feature Engineering &amp; NLP",
        "subtitle": "Transformaciones, embeddings, topics y enrichment externo",
    },
    {
        "file": "04_models.ipynb",
        "html": "nb04_models.html",
        "label": "NB04: Models",
        "title": "04 - Modelos",
        "subtitle": "Lead scoring, clustering y SHAP",
    },
    {
        "file": "05_mlops.ipynb",
        "html": "nb05_mlops.html",
        "label": "NB05: MLOps",
        "title": "05 - MLOps &amp; Deployment",
        "subtitle": "API, Docker, Airflow, MLflow y monitorizaci&oacute;n",
    },
]

SIDEBAR_CSS = """<style id="sidebar-nav-css">
#sidebar-nav {
    all: initial;
    position: fixed; top: 0; left: 0; width: 200px; height: 100vh;
    background: #141413; color: #B5B5AE; padding: 1.5rem 1.2rem;
    box-sizing: border-box; z-index: 9999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.82rem; line-height: 1.4;
    display: flex; flex-direction: column; overflow-y: auto;
}
#sidebar-nav * {
    all: unset;
    box-sizing: border-box;
}
#sidebar-nav .sidebar-title {
    display: block;
    font-family: Georgia, serif; font-size: 1.1rem; color: #FAFAF7;
    font-weight: 700; margin-bottom: 0.3rem;
}
#sidebar-nav .sidebar-subtitle {
    display: block;
    font-size: 0.7rem; color: #8B8B85; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 1rem;
}
#sidebar-nav .sidebar-divider {
    display: block;
    height: 2px; background: linear-gradient(90deg, #FFC630, #141413);
    margin: 0 0 1rem 0; border: none;
}
#sidebar-nav a {
    display: block; color: #B5B5AE; text-decoration: none;
    padding: 0.35rem 0.6rem; margin: 0.1rem 0; border-radius: 4px;
    border-left: 2px solid transparent; font-size: 0.82rem;
    transition: all 0.15s; cursor: pointer;
}
#sidebar-nav a:hover {
    color: #FAFAF7; border-left-color: #FFC630; background: rgba(255,198,48,0.08);
}
#sidebar-nav a.sidebar-active {
    color: #FFC630; font-weight: 600; border-left-color: #FFC630;
    background: rgba(255,198,48,0.12);
}
#sidebar-nav .sidebar-footer {
    display: block;
    margin-top: auto; padding-top: 1rem; font-size: 0.7rem; color: #6B6B66;
}
body { margin-left: 200px !important; }
</style>"""

RAONA_CSS = """<style id="raona-overrides">
body { background-color: #FAFAF7 !important; }
#site { background-color: #FAFAF7 !important; }
.jp-Notebook { max-width: 960px; margin: 0 auto; padding: 0 2rem; }
.nb-header {
    padding: 3rem 0 2rem 0; border-bottom: 1px solid #E8E8E3; margin-bottom: 2rem;
}
.nb-header .nb-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #8B8B85; margin-bottom: 0.5rem;
}
.nb-header .nb-title {
    font-family: Georgia, serif; font-size: 2.4rem; font-weight: 700;
    color: #141413; line-height: 1.15; margin-bottom: 0.5rem;
}
.nb-header .nb-subtitle {
    font-size: 1.1rem; color: #6B6B66; max-width: 700px;
}
.nb-footer {
    margin-top: 3rem; padding: 2rem 0; border-top: 1px solid #E8E8E3;
    font-size: 0.8rem; color: #8B8B85; text-align: center;
}
.jp-MarkdownOutput h1, .jp-MarkdownOutput h2, .jp-MarkdownOutput h3 {
    font-family: Georgia, serif !important; color: #141413 !important;
}
.jp-MarkdownOutput h1 { font-size: 2rem !important; margin-top: 3rem !important; }
.jp-MarkdownOutput h2 {
    font-size: 1.5rem !important; margin-top: 2.5rem !important;
    padding-top: 1.5rem !important; border-top: 1px solid #E8E8E3 !important;
}
.jp-MarkdownOutput h3 { font-size: 1.2rem !important; margin-top: 1.5rem !important; }
.jp-MarkdownOutput p { color: #3D3D3A; line-height: 1.7; }
.jp-MarkdownOutput strong { color: #141413; }
.jp-MarkdownOutput table, .jp-RenderedHTMLCommon table {
    border-collapse: collapse; width: 100%; max-width: 800px;
    margin: 1.5rem 0; font-size: 0.9rem;
}
.jp-MarkdownOutput th, .jp-RenderedHTMLCommon th {
    background: #F0F0EB; font-weight: 600; text-align: left;
    padding: 0.8rem 1rem; border-bottom: 2px solid #E8E8E3;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8B8B85;
}
.jp-MarkdownOutput td, .jp-RenderedHTMLCommon td {
    padding: 0.7rem 1rem; border-bottom: 1px solid #E8E8E3;
}
.jp-MarkdownOutput tr:hover, .jp-RenderedHTMLCommon tr:hover {
    background-color: #FFF3D1;
}
.jp-InputArea-editor { border-radius: 6px; border: 1px solid #E8E8E3 !important; }
.jp-OutputArea-output pre { font-size: 0.85rem; }
.jp-OutputArea-output img { max-width: 100%; height: auto; border-radius: 4px; }
.jp-RenderedHTMLCommon, .jp-MarkdownOutput {
    overflow-x: auto !important;
}
.jp-OutputArea-output > div {
    overflow-x: auto !important;
}
table.dataframe {
    display: block; overflow-x: auto; white-space: nowrap; max-width: 100%;
    font-size: 0.82rem; border: 1px solid #E8E8E3; border-radius: 6px;
    border-collapse: separate; border-spacing: 0;
}
table.dataframe th {
    background: #F0F0EB !important; position: sticky; top: 0;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #6B6B66; padding: 0.6rem 0.8rem;
    border-bottom: 2px solid #E8E8E3; white-space: nowrap;
}
table.dataframe td {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid #E8E8E3; white-space: nowrap;
}
table.dataframe tr:hover td {
    background-color: #FFF3D1;
}
.jp-MarkdownOutput table {
    display: block; overflow-x: auto; white-space: nowrap;
}
</style>"""


def build_sidebar_html(active_html):
    links = []
    links.append('  <a href="index.html"{}>{}</a>'.format(
        ' class="sidebar-active"' if active_html == "index.html" else "",
        "Data Story"
    ))
    for nb in NOTEBOOKS:
        cls = ' class="sidebar-active"' if nb["html"] == active_html else ""
        links.append(f'  <a href="{nb["html"]}"{cls}>{nb["label"]}</a>')
    # App Demo link (external, opens in new tab)
    links.append('  <hr class="sidebar-divider">')
    links.append('  <a href="https://raona-lead-scoring.streamlit.app" target="_blank">App Demo &#8599;</a>')

    return """<nav id="sidebar-nav">
<div class="sidebar-title">Raona Lead Scoring</div>
<div class="sidebar-subtitle">TFM - Nuclio Digital School</div>
<hr class="sidebar-divider">
{links}
<div class="sidebar-footer">G03<br>Marzo 2026</div>
</nav>""".format(links="\n".join(links))


def build_header(nb_info):
    return """<div class="nb-header">
    <div class="nb-label">Raona Lead Scoring &middot; TFM Nuclio Digital School</div>
    <div class="nb-title">{title}</div>
    <div class="nb-subtitle">{subtitle}</div>
    <hr>
</div>""".format(**nb_info)


NB_FOOTER = '<div class="nb-footer">G03 &middot; Nuclio Digital School &middot; Data Science Master &middot; Marzo 2026</div>'


def clean_outputs(nb):
    """Remove warnings from stderr and sanitize local paths in all outputs."""
    warning_patterns = [
        "RuntimeWarning", "UserWarning", "FutureWarning", "DeprecationWarning",
        "NotOpenSSLWarning", "ConvergenceWarning", "TqdmWarning",
        "warnings.warn(", "Warning:",
    ]
    # Local path prefixes to strip (most specific first)
    gdrive_base = "Library/CloudStorage/GoogleDrive-adriana.caballero@gmail.com/.shortcut-targets-by-id/1LzrxzfxIAZukyDLfOvioF7Z2ESoOXZGz/TFM/acaballero/"
    project_prefix = "/Users/acaballito/" + gdrive_base + "TFM_deliverables/"
    parent_prefix = "/Users/acaballito/" + gdrive_base
    home_prefix = "/Users/acaballito/"
    lib_prefix = "Library/Python/3.9/lib/python/site-packages/"
    # Also handle after file:// or ./ replacement
    gdrive_base_deliverables = gdrive_base + "TFM_deliverables/"

    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.outputs:
            continue
        cleaned_outputs = []
        for out in cell.outputs:
            # Filter stderr streams that are pure warnings
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = out.get("text", "")
                # Remove lines that contain warning patterns
                lines = text.split("\n")
                kept = [l for l in lines if not any(wp in l for wp in warning_patterns)]
                remaining = "\n".join(kept).strip()
                if not remaining:
                    continue  # drop entirely
                out["text"] = remaining + "\n"

            # Sanitize local paths in all text content
            def sanitize_path(text):
                text = text.replace(project_prefix, "./")
                text = text.replace(parent_prefix, "./")
                text = text.replace(home_prefix, "./")
                text = text.replace(gdrive_base_deliverables, "./")
                text = text.replace(gdrive_base, "./")
                text = text.replace(lib_prefix, "")
                return text

            if out.get("output_type") == "stream":
                out["text"] = sanitize_path(out.get("text", ""))
            if hasattr(out, "data"):
                for mime in list(out.data.keys()):
                    if isinstance(out.data[mime], str):
                        out.data[mime] = sanitize_path(out.data[mime])
            if hasattr(out, "text") and isinstance(out.text, str):
                out.text = sanitize_path(out.text)

            cleaned_outputs.append(out)
        cell.outputs = cleaned_outputs
    return nb


def _decode_bdata(obj):
    """Recursively decode Plotly binary-encoded arrays (bdata+dtype) to plain lists."""
    import base64
    import numpy as np

    DTYPE_MAP = {
        "f8": np.float64, "f4": np.float32,
        "i4": np.int32, "i2": np.int16, "i1": np.int8,
        "u4": np.uint32, "u2": np.uint16, "u1": np.uint8,
    }

    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            raw = base64.b64decode(obj["bdata"])
            dt = DTYPE_MAP.get(obj["dtype"], np.float64)
            arr = np.frombuffer(raw, dtype=dt)
            return [None if np.isnan(v) else v for v in arr] if np.issubdtype(dt, np.floating) else arr.tolist()
        return {k: _decode_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata(item) for item in obj]
    return obj


def plotly_to_html_outputs(nb):
    """Convert Plotly JSON outputs to text/html so nbconvert can render them."""
    plotly_js_included = False
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.outputs:
            continue
        for out in cell.outputs:
            if not hasattr(out, "data"):
                continue
            if "application/vnd.plotly.v1+json" in out.data:
                fig_dict = out.data["application/vnd.plotly.v1+json"]
                if isinstance(fig_dict, str):
                    fig_dict = json.loads(fig_dict)
                fig_dict = _decode_bdata(fig_dict)
                fig = pio.from_json(json.dumps(fig_dict), output_type="Figure")
                html_str = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs="cdn" if not plotly_js_included else False,
                )
                plotly_js_included = True
                out.data["text/html"] = html_str
                del out.data["application/vnd.plotly.v1+json"]
    return nb


def convert_notebook(nb_info):
    ipynb = NB_DIR / nb_info["file"]
    out_html = REPORT_DIR / nb_info["html"]

    if not ipynb.exists():
        print(f"  SKIP: {ipynb} not found")
        return

    # Step 1: nbconvert to HTML via Python API
    print(f"  Converting {nb_info['file']}...")
    nb = nbformat.read(str(ipynb), as_version=4)
    nb = clean_outputs(nb)
    nb = plotly_to_html_outputs(nb)
    exporter = HTMLExporter()
    html, resources = exporter.from_notebook_node(nb)

    # Inject CSS before </head>
    css_block = SIDEBAR_CSS + "\n" + RAONA_CSS
    html = html.replace("</head>", css_block + "\n</head>")

    # Inject sidebar + header after <body...>
    sidebar = build_sidebar_html(nb_info["html"])
    header = build_header(nb_info)
    # Match <body ...>
    body_match = re.search(r"(<body[^>]*>)", html)
    if body_match:
        insert_pos = body_match.end()
        html = html[:insert_pos] + "\n" + sidebar + "\n\n" + header + "\n<main>\n" + html[insert_pos:]

    # Inject footer before </body>
    html = html.replace("</body>", NB_FOOTER + "\n</main>\n</body>")

    # Replace "Adriana Caballero" with "G03"
    html = html.replace("Adriana Caballero", "G03")

    # Sanitize local paths in final HTML (most specific first)
    gdrive = ("Library/CloudStorage/GoogleDrive-adriana.caballero@gmail.com/"
              ".shortcut-targets-by-id/1LzrxzfxIAZukyDLfOvioF7Z2ESoOXZGz/TFM/acaballero/")
    html = html.replace("/Users/acaballito/" + gdrive + "TFM_deliverables/", "./")
    html = html.replace("/Users/acaballito/" + gdrive, "./")
    html = html.replace("/Users/acaballito/", "./")
    html = html.replace(gdrive + "TFM_deliverables/", "./")
    html = html.replace(gdrive, "./")

    out_html.write_text(html, encoding="utf-8")
    size_kb = out_html.stat().st_size / 1024
    print(f"  Done: {nb_info['html']} ({size_kb:.0f} KB)")


def main():
    print("Converting notebooks to HTML...")
    for nb_info in NOTEBOOKS:
        convert_notebook(nb_info)
    print("\nAll done.")


if __name__ == "__main__":
    main()
