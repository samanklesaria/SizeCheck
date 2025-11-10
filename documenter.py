import importlib.metadata
import pdoc
import os
import tomlkit
import ghp_import

def recursive_htmls(mod):
    yield mod.name, mod.html(show_source_code=False)
    for submod in mod.submodules():
        yield from recursive_htmls(submod)

def generate_documentation(main_module):
    def index_content(mod):
        return f"""
        <!doctype html>
        <html>
            <head>
                <meta http-equiv="refresh" content="0; url=./{mod}.html" />
            </head>
        </html>
        """
    version = importlib.metadata.version(main_module)
    context = pdoc.Context()
    mod = pdoc.Module(main_module, context=context)
    pdoc.link_inheritance(context)
    os.makedirs(f"docs/{version}", exist_ok=True)
    if not os.path.exists(f"docs/{version}/index.html"):
        with open(f"docs/{version}/index.html", "w") as f:
            f.write(index_content(main_module))
    for module_name, html in recursive_htmls(mod):
        with open(f"docs/{version}/{module_name}.html", "w") as f:
            f.write(html)
    if os.path.exists("docs/latest"):
        os.remove("docs/latest")
    os.symlink(os.path.abspath(f"docs/{version}"), "docs/latest", target_is_directory=True)
    with open("docs/index.html", "w") as f:
        f.write(index_content(f"{version}/{main_module}"))
    ghp_import.ghp_import("docs", push=True, followlinks=True)

if __name__ == "__main__":
    with open("pyproject.toml", "r") as pyproject:
        file_contents = pyproject.read()
    generate_documentation(tomlkit.parse(file_contents)["project"]["name"])
