from jinja2 import Template


def render_template(template_content: str, context: dict) -> str:
    """
    Render a pipeline template with the provided context.

    Args:
        template_content (str): The content of the template.
        context (dict): The context to render the template with.

    Returns:
        str: The rendered template as a string.
    """
    template = Template(template_content)
    return template.render(context)
