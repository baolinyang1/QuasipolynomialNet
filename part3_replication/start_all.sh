for dataset_idx in {0..13}
do
  dataset_number=$((dataset_idx + 1))
  echo "starting dataset$dataset_number"
  screen -dmS "dataset$dataset_number" bash -c "python3 -u replication-aio.py -d $dataset_idx -s 100  2>&1 | tee log/dataset${dataset_number}_$(date +'%Y%m%d_%H%M%S').txt"

done




# rm -rf exp && rm -rf exp.zip && rm -rf log/* && rm -rf data/* && rm -rf experiments_*

# rm -rf exp && rm -rf exp.zip && mkdir -p exp && cp experiments_* exp && zip -r exp.zip exp



# python3 tocsv.py data/experiments/ result.csv

# timestamp=$(date +"%Y%m%d_%H%M%S") && cd .. && zip -r project_$timestamp.zip part3 && cd part3