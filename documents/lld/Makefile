PUML_FILES := $(wildcard diagrams/*.puml)
PNG_FILES  := $(patsubst diagrams/%.puml,img/%.png,$(PUML_FILES))

PLANTUML   ?= plantuml

all: img $(PNG_FILES) main.pdf

img:
	mkdir -p img

img/%.png: diagrams/%.puml
	$(PLANTUML) -o ../img $<

main.pdf: main.tex sections/*.tex $(PNG_FILES)
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.toc *.out main.pdf
	rm -rf img/

.PHONY: all clean
