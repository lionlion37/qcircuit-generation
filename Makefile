SHELL := /usr/bin/bash
.ONESHELL:

all:
	make codex
	make qc-env

codex:
	curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
	source $(HOME)/.nvm/nvm.sh
	nvm install node
	nvm use node >/dev/null
	npm i -g @openai/codex

qc-env:
	pip install -r requirements.txt
	pip install -r quditkit-main_schmidt/requirements.txt
	pip install -e quditkit-main_schmidt
	git config --global user.email "lidungl@archimedia.at"
	git config --global user.name "Lion Dungl"

