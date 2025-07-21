import pandas as pd
from jinja2 import Environment, FileSystemLoader

def generate_report(stats: dict, out_path: str):
    env = Environment(loader=FileSystemLoader('templates'))
    tmpl = env.get_template('report.html')
    html = tmpl.render(stats=stats)
    with open(out_path, 'w') as f:
        f.write(html)
