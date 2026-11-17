# Requires a TeX distribution (texlive-full or similar). This machine has none;
# compile on Overleaf or install one locally.
MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -C
	rm -f sections/*.aux

.PHONY: all clean
