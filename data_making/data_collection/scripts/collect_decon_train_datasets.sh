cd ..

if [ ! -d "data" ]; then
    mkdir "data"
    echo "Folder data created."
else
    echo "Folder data already exists."
fi

if [ ! -d ".cache" ]; then
    mkdir ".cache"
    echo "Folder .cache created."
else
    echo "Folder .cache already exists."
fi

python collect_data.py