mkdir ./archive/images_resized
mogrify -resize 1280x720 ./archive/images/* -path ./archive/images_resized