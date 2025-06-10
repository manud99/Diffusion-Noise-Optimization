if [ -d "save/mdm_avg_dno" ]; then
    echo "Error: Directory save/mdm_avg_dno already exists."
    exit 1
fi

mkdir -p save
wget -O save/mdm_avg_dno.zip https://polybox.ethz.ch/index.php/s/ZiXkIzdIsspK2Lt/download
unzip save/mdm_avg_dno.zip -d save/
rm save/mdm_avg_dno.zip
