# Minimal makefile for pdoc documentation
#

# You can set these variables from the command line.
BINBUILD   = pdoc
PROJ     = ../micarrayx
BUILDDIR      = .

.phony: makefile

%: makefile
	@$(BINBUILD) ${PROJ} --html -o "$(BUILDDIR)" --force

