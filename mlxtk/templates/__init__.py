import jinja2

TEMPLATE_ENV = jinja2.Environment(
    loader=jinja2.PackageLoader("mlxtk", "templates"))


def get_template(name: str) -> jinja2.Template:
    return TEMPLATE_ENV.get_template(name)
