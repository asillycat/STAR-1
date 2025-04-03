cd ..

if [ ! -d "datasets" ]; then
    mkdir "datasets"
    echo "Folder datasets created."
else
    echo "Folder datasets already exists."
fi

python load_test_datasets.py