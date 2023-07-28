echo "Uninstall old PAMI version"
pip3 uninstall -y pami

echo "Running setup"
python3 setup.py sdist bdist_wheel

echo "Uploading to test repository"
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

echo "Wait for 5 minute to update the repository"
sleep 60

echo "installing PAMI from the testPYPI"
python3 -m pip3 install --index-url https://test.pypi.org/simple/ --no-deps pami

echo "Uploading PAMI to main PYPI repository"
python3 -m twine upload dist/*

echo "Deleting unnecessary files"
rm -rf dist/ pami.egg-info/ build/


echo "Completed."
