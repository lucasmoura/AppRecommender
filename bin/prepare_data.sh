SCRIPT_PATH=$(readlink -f "$0")
SRC_DIR=$(dirname "$SCRIPT_PATH")
#PATH_DIR=${1-/vagrant}
PATH_DIR=~

echo "$SRC_DIR"

echo "Deleting folder: $PATH_DIR/.app-recommender"
rm -rf $PATH_DIR/.app-recommender

echo "Creating folder: $PATH_DIR/.app-recommender"
mkdir $PATH_DIR/.app-recommender

echo "Creating folder: $PATH_DIR/.app-recommender/filters"
cd $PATH_DIR/.app-recommender
mkdir filters

cd $SRC_DIR

echo ""
echo "Creating filter: debtags"
echo "Running command: $ $SRC_DIR/get_tags.sh > $PATH_DIR/.app-recommender/filters/debtags"
./get_tags.sh > $PATH_DIR/.app-recommender/filters/debtags

echo ""
echo "Creating filter desktopapps"
echo "Running command: $ $SRC_DIR/get_axipkgs.py > $PATH_DIR/.app-recommender/filters/desktopapps"
python get_axipkgs.py > $PATH_DIR/.app-recommender/filters/desktopapps

echo ""
echo "Indexing axi sample: desktopapps"
echo "Running command: $ $SRC_DIR/indexer_axi.py sample $PATH_DIR/.app-recommender/filters/desktopapps"
python indexer_axi.py sample $PATH_DIR/.app-recommender/filters/desktopapps

