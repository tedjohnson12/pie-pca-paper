compile_dir=./.showyourwork/compile
dest_dir=./aasjournal

rm -r $dest_dir
mkdir $dest_dir
cp -r $compile_dir/* $dest_dir

cd $dest_dir
mv figures/* .
rm figures/.gitignore
rmdir figures

mv output/* .
rm output/.gitignore
rmdir output

mv ms.tex aasjournal.tex

sed -i -e "s/figures\///g" aasjournal.tex
sed -i -e "s/output\///g" aasjournal.tex