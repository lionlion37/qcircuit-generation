env:
	pip install -r requirements.txt
	pip install -r quditkit-main_schmidt/requirements.txt
	pip install -e quditkit-main_schmidt
	git config --global user.email "lidungl@archimedia.at"
	git config --global user.name "Lion Dungl"
