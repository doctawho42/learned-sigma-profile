# Manuscript sources

The LaTeX sources of the article and of its Supporting Information are added here when the
article is submitted. Until then this directory holds only the places the analysis scripts write
to: `figs/` for the figures and `si_tables/` for the Supporting Information tables.

Two consequences while the sources are absent, both of them explicit rather than silent:

* `tests/test_manuscript_numbers_bind.py` skips, with the reason printed. It binds each number
  printed in the article to the deposit it is read from, so it needs the text to read.
* `tools/check_si_contract.py` skips for the same reason. It reads the article's Supporting
  Information section as a contract and checks the Supporting Information against it.

Everything else runs: the other tests, the other three checks, and every analysis script.
