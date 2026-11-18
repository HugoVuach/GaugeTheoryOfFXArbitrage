"""Append the research extension to the preserved report, as standalone TeX."""
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parent


def assemble(output):
    base = (ROOT / 'source/legacy_standalone.tex').read_text()
    extra = (ROOT / 'source/extensions.tex').read_text()

    def table(match):
        return (ROOT / match.group(1)).read_text().rstrip()

    extra = re.sub(r'\\input\{(results/extension/[a-z_]+\.tex)\}', table, extra)

    def figure(match):
        width, name = match.groups()
        plot = (ROOT / 'figures' / (name + '.pgf')).read_text()
        plot = plot[plot.index(r'\begingroup'):].rstrip()
        return (r'\begin{lrbox}{\plotbox}' + '%\n' + plot + '\n'
                + r'\end{lrbox}' + '%\n'
                + rf'\resizebox{{{width}}}{{!}}{{\usebox{{\plotbox}}}}')

    extra, count = re.subn(r'\\includegraphics\[width=([^\]]+)\]\{(extension_[a-z_]+)\.pdf\}', figure, extra)
    if count != 6 or r'\input{' in extra or r'\includegraphics' in extra:
        raise ValueError('Six new vector figures and all generated tables must be embedded')
    if base.count(r'\end{document}') != 1:
        raise ValueError('Unexpected preserved document structure')
    final = base.replace(r'\end{document}', extra + '\n' + r'\end{document}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(final)
    print(f'Standalone report: {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'Note_Jauge_Arbitrage_FX_3_Experiences.tex')
    parser.add_argument('--skip-plots', action='store_true', help='Use existing vector figures and generated tables')
    args = parser.parse_args()
    if not args.skip_plots:
        import extension_report
        extension_report.build_figures(ROOT / 'results/extension', ROOT / 'figures')
        extension_report.build_tables(ROOT / 'results/extension')
    assemble(args.output)
