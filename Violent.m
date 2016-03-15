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

% For each frame in the video:
% Write the still grayscale frame to disk.  
% Binarize the difference between the current frame and previous frame.  Write the binarized difference to disk.
% Extract relevant features.
while hasFrame(v)
    
    % Read in the frame.  Scale it, grasyscale filter it, and determine gray threshold for binarizing the image.
    thisFrame = readFrame(v);
    imagesc(thisFrame);
    thisGray = rgb2gray(thisFrame);
    thisLevel = graythresh(thisGray);
    
    % Dump the grayscale image.
    opBaseFileName = sprintf('%3.3d.png', k);
    opFullFileName = fullfile(opFolder, opBaseFileName);
    imwrite(thisGray, opFullFileName, 'png');
    
    if(k > 1)
        
        % Load the file at the previous index k - 1.
        prevBaseFileName = sprintf('%3.3d.png', k - 1);
        prevFullFileName = fullfile(opFolder, prevBaseFileName);
        prevImage = imread(prevFullFileName);
        
        % Binarize the absolute difference across both frames.
        difference = imabsdiff(thisGray, prevImage);
        thisBW = im2bw(difference, thisLevel);
        
        % Write the binarized difference file.
        diffBaseFileName = sprintf('Diff%3.3d.png', k);
        diffFullFileName = fullfile(diffFolder, diffBaseFileName);
        imwrite(thisBW, diffFullFileName, 'png');
        
        % Read centroids.
        s = regionprops(thisBW, 'centroid');
        centroids = cat(1, s.Centroid);
        plot(centroids( :, 1), centroids(:, 2), 'b*');
        
        % Read perimeter and area.
        p = regionprops(thisBW, 'perimeter');
        a = regionprops(thisBW, 'area');
        % e = 
    end

    k = k+1;
end

%implay(videoFile);


