#!/usr/bin/env bash
# Public-domain novels from Project Gutenberg via the GITenberg mirror.
mkdir -p corpus && cd corpus
for b in Pride-and-Prejudice_1342/master/1342.txt A-Tale-of-Two-Cities_98/master/98.txt Moby-Dick--Or-The-Whale_2701/master/2701.txt Great-Expectations_1400/master/1400.txt Emma_158/master/158.txt Jane-Eyre_1260/master/1260.txt Middlemarch_145/master/145.txt Little-Women_514/master/514.txt Dracula_345/master/345.txt Frankenstein_84/master/84.txt; do
  curl -sL -o "$(basename $b)" "https://raw.githubusercontent.com/GITenberg/$b"; done
