"""Tasks for aiding in notetaking."""
import pathlib

import invoke


@invoke.task(
    help={"chapter": "The chapter number", "name": "The new name of the file."}
)
def screenshot(context, chapter, name):
    """Move screenshot to the images directory and copy relative path."""
    home_dir = pathlib.Path.home()
    screenshot_dir = next((home_dir / pathlib.Path("Desktop")).glob("*.png"))
    doc_dir = home_dir / pathlib.Path.cwd() / pathlib.Path("docs", "hackers")
    relative_new_dir = pathlib.Path("images", chapter)
    (doc_dir / relative_new_dir).mkdir(parents=True, exist_ok=True)
    relative_new_filepath = relative_new_dir / pathlib.Path(name).with_suffix(".png")
    context.run(
        f"mv '{screenshot_dir}' '{doc_dir / relative_new_filepath}'"
    )
    context.run(f"printf '{relative_new_filepath}' | pbcopy")
