"""Append the actual eight-currency campaign, embedding all new vector plots."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def assemble():
    base=(ROOT/'source/previous_50_pages.tex').read_text()
    extra=(ROOT/'source/long_period.tex').read_text()
    extra=re.sub(r'\\input\{(results/long_period/[a-z_]+\.tex)\}',lambda m:(ROOT/m.group(1)).read_text().rstrip(),extra)
    def figure(match):
        width,name=match.groups();plot=(ROOT/'figures'/(name+'.pgf')).read_text();plot=plot[plot.index(r'\begingroup'):].rstrip()
        return (r'\begin{lrbox}{\plotbox}'+'%\n'+plot+'\n'+r'\end{lrbox}'+'%\n'+rf'\resizebox{{{width}}}{{!}}{{\usebox{{\plotbox}}}}')
    extra,count=re.subn(r'\\includegraphics\[width=([^\]]+)\]\{(emp_[a-z_]+)\.pdf\}',figure,extra)
    if count!=6 or r'\input{' in extra or r'\includegraphics' in extra:raise ValueError('All six figures and every new table must be embedded')
    if base.count(r'\end{document}')!=1:raise ValueError('Unexpected preserved report structure')
    output=ROOT/'Note_Jauge_Arbitrage_FX_3_Experiences.tex';output.write_text(base.replace(r'\end{document}',extra+'\n'+r'\end{document}'))
    print('Standalone LaTeX',output,output.stat().st_size)

if __name__=='__main__':assemble()
