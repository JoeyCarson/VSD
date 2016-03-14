violentVideoFile = 'C:\Program Files (x86)\MATLAB\R2015b\toolbox\matlab\audiovideo\video.avi';
v = VideoReader(violentVideoFile);
frames = v.CurrentTime;
opFolder = fullfile(cd, 'Images');
diffFolder = fullfile(cd, 'Differences');
if ~exist(opFolder, 'dir')
    mkdir(opFolder);
end
if ~exist(diffFolder, 'dir')
    mkdir(diffFolder);
end
k = 1;
while hasFrame(v)
    thisFrame = readFrame(v);
    imagesc(thisFrame);
    thisGray = rgb2gray(thisFrame);
    thisLevel = graythresh(thisGray);
    opBaseFileName = sprintf('%3.3d.png', k);
    opFullFileName = fullfile(opFolder, opBaseFileName);
    imwrite(thisGray, opFullFileName, 'png');
    if(k > 1)
        prevBaseFileName = sprintf('%3.3d.png', k - 1);
        prevFullFileName = fullfile(opFolder, prevBaseFileName);
        prevImage = imread(prevFullFileName);
        difference = imabsdiff(thisGray, prevImage);
        thisBW = im2bw(difference, thisLevel);
        diffBaseFileName = sprintf('Diff%3.3d.png', k);
        diffFullFileName = fullfile(diffFolder, diffBaseFileName);
        imwrite(thisBW, diffFullFileName, 'png');
        s = regionprops(thisBW, 'centroid');
        centroids = cat(1, s.Centroid);
        plot(centroids( :, 1), centroids(:, 2), 'b*');
        
        p = regionprops(thisBW, 'perimeter');
        a = regionprops(thisBW, 'area');
%        e = 
    end
    k = k+1;
end

%implay(videoFile);


