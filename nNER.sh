#!/bin/sh
python evaluate_model.py > classification_binary.txt
python EvaluateModelPart2.py > classification_3class.txt
python EvaluateModelPart3.py > classification_5class.txt
