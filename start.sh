#!/bin/bash

ollama serve &

sleep 5

python app.py

