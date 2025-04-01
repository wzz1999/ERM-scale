function [coordinates,himage] = neuronsSpatialDistribution2(k,A,ObjRecon)
% Syntax: neuronsSpatialDistribution(k,A,ObjRecon)
% Input:
%         k: the indices of ROIs
%         A: footprint of ROIs(You can load the variable 'A3' from the segmentation results 'Coherence3.mat')
%         ObjRecon: pick one frame as background

% Output:
%         coordinates: the coordinates of selected neurons
%         himage: figure handle

% 
% Long description
%   Draw the spatial distribution of neurons. 

% July 21 updates:
% Add Line 75 for saving plots
% now dx = d1; dy = d2; dz = dz, no longer need shift.
    % mycolor = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'];
    mycolor = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
    mycolor = mycolor([7,4,1,5,6,3],:);
%     my_color = hsv(length(k));
    my_color = [255,165,0; 0,255,127; 0,191,255; 71,2,174; 255,20,147]/256;
    d1 = size(ObjRecon,1);
    d2 = size(ObjRecon,2);
    d3 = size(ObjRecon,3);
    dx = d1;
    dy = d2;
    dz = d3;
%     
    if (d1==dx) && (d2==dy) && (d3==dz)
        x_shift = 0;
        y_shift = 0;
        z_shift = 0;
    else
        x_shift = 140; % If the registered image is smaller than the original one, there should be a shift.
        y_shift = 0;
        z_shift = 60;
    end

    MIP = [max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];    
    MIP(MIP > prctile(MIP(:),99)) = prctile(MIP(:),99);
%     MIP = MIP/prctile(MIP(:),96);

    MIP = single(repmat(MIP,1,1,3)); % convert to 3 channels RGB image
    MIP = MIP/100;
    x_c = zeros(1,length(k));
    y_c = zeros(1,length(k));
    z_c = zeros(1,length(k));
    for index_k=1:length(k) % for each neuron, draw its contour
        temp = find(full(A(:,k(index_k))));
        [x,y,z] = ind2sub([dx,dy,dz],temp);
        x = x - x_shift;
        y = y - y_shift;
        z = z - z_shift;
        x_c(index_k) = round(mean(x));
		y_c(index_k) = round(mean(y));
		z_c(index_k) = round(mean(z));
        temp = zeros(size(MIP,1),size(MIP,2)); % the contour of the neuron on MIP image
        temp_index = sub2ind(size(MIP),x,y);
        temp(temp_index) = 1;
        temp_index = sub2ind(size(MIP),x,z+d2);
        temp(temp_index) = 1;
        temp_index = sub2ind(size(MIP),z+d1,y);
        temp(temp_index) = 1;
        temp = bwperim(temp);
        temp = imfill(temp,'holes');
        R = squeeze(MIP(:,:,1));
        G = squeeze(MIP(:,:,2));
        B = squeeze(MIP(:,:,3));
        if length(k) <= 6
            R(temp) = mycolor(index_k,1);
            G(temp) = mycolor(index_k,2);
            B(temp) = mycolor(index_k,3);
        else
            R(temp) = my_color(mod(index_k,5)+1,1); % select from 5 color
            G(temp) = my_color(mod(index_k,5)+1,2);
            B(temp) = my_color(mod(index_k,5)+1,3);
        end

        MIP(:,:,1) = R;
        MIP(:,:,2) = G;
        MIP(:,:,3) = B;
    end

    %% display
    himage=imshow(MIP,[]);
%     for index_k=1:length(k) % for each neuron, add a text
%         % color_k = squeeze(ind2rgb(mod(k(index_k),64),hsv));
%         text(y_c(index_k),x_c(index_k),int2str(k(index_k)),'Color','r');
%     end
%     hold on;
%     imshow(MIP2);
    

 for i=1:length(k)
 coordinates{i}=[x_c(i),y_c(i),z_c(i)];
 end

end