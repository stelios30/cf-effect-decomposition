#!/bin/bash
cp -r . ~/Desktop/counterfactual_effect_decomposition_in_ma_sequential_decision_making
cd ~/Desktop/counterfactual_effect_decomposition_in_ma_sequential_decision_making

rm -rf .git

find . -regex '^.*\(__pycache__\|\.py[co]\)$' -delete
rm .env
rm -rf .pytest_cache
rm -rf .vscode
rm -rf ./ase

rm notebooks/playground.ipynb
rm notebooks/grid/.gitkeep

rm -rf plots/grid/*.pdf
rm -rf plots/sepsis/*.pdf

rm -rf results/grid/*
rm -rf results/sepsis/*

cd ~/Desktop
zip -r ~/Desktop/code.zip counterfactual_effect_decomposition_in_ma_sequential_decision_making
cd ~/Desktop
rm -rf ~/Desktop/counterfactual_effect_decomposition_in_ma_sequential_decision_making
echo "Done compressing code, see ~/Desktop/counterfactual_effect_decomposition_in_ma_sequential_decision_making.zip"
