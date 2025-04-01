


function Fig_experimental_observation(matroot)
addpath(genpath('C:\Users\Public\code\Fish-Brain-Behavior-Analysis\code\data_analysis_Weihao\functions'));
addpath(genpath('C:\Users\Public\code\Fish-Brain-Behavior-Analysis\code\RG\functions'));

isubject = '201106';
dir = getFishPath('onedrive',isubject);
dir_nut = getFishPath('activity',isubject);
load([dir,'\spike_OASIS.mat'],'sMatrix_total');
load([dir,'\Judge.mat'],'Judge');
load([dir,'\Judge2.mat'],'Judge2');
% 
FR = sMatrix_total(Judge2,:);
tf = sum(FR,2) > 0;
FR = FR(tf,:);
FR = normalize_activity(FR,'mean');
[nC,nT] = size(FR);
load([dir,'\pca_result.mat'],'U');

% path1 = 'D:\OneDrive - HKUST Connect\data\stringer_et_al\stringer_neuropixels\'
% % isubject = 'Krebs';
% % isubject = 'Robbins';
% isubject = 'Waksman';
% load(fullfile(path1,[isubject,'_FR.mat']));
% tf = sum(FR,2) > 0;
% FR = double(FR(tf,:));
% FR = normalize_activity(FR,'mean');
%%
close all;
figX = 4; % size of the page
figY = 6;
default_figure([1 1 figX figY]);
%%

yh = .05;
xh = .02;

dex=2;
clf;
clear hs;
cm = colormap('winter');
load('cdat.mat');

ifr =1;
cgray=colormap('gray');

ncomps = 4;
cm = cm(round(linspace(1,size(cm,1),ncomps)),:);

nPCs = 4;
cpc = max(0,colormap('spring'));
cpc = cpc(1:32,:);
cpc = cpc(round(linspace(1,size(cpc,1),nPCs)),:);
spc = [-1 1 -1 -1];
ndat = size(cdat,1);

i = 0;

i=i+1;
% [ny,nx,~]=size(exframes);
% hs{i} = my_subplot(4,5,1,[.65 .6]);
% hs{i}.Position(1) = hs{i}.Position(1)+.01;
% hs{i}.Position(2) = hs{i}.Position(2)+.01;
% hold all;
% imagesc([1:nx]-50,[1:ny]-50,exframes(:,:,ifr));
% imagesc([1:nx],[1:ny],exframes(:,:,ifr+1));
% text(-.1,0.22,'t','fontsize',6);
% text(0,0,'t+1','fontsize',6);
% colormap(gca,cgray);
% set(gca,'ydir','reverse');
% axis image;
% axis off;

%% segmentation of ROI


hs{i} = my_subplot(2,2,i,[.85 .55]); % i is the ith panel, [.65 .5] resize the panel
hs{i}.Position(2)=hs{i}.Position(2)+0.01;
hs{i}.Position(1) = hs{i}.Position(1)+0.05;
hold all;
% load([dir_nut,'\Coherence3.mat'],'A3');
% A3 = A3(:,Judge);
% A3 = A3(:,Judge2);
% i_rand = randsample(size(A3,2),size(A3,2));
% 
% load('affine_X290.mat','ObjRecon');
% figure;
% neuronsSpatialDistribution2(i_rand,A3,ObjRecon);
fig_tmp = openfig('segmentation.fig','invisible'); % open figure
ax_old = findobj(fig_tmp, 'type', 'axes');
copyobj(ax_old.Children, hs{i});
close(fig_tmp);
title("Segmentation of ROIs");
axis off;
axis equal;
set(gca, 'YDir','reverse')
% close(fig_tmp);



% ax_old.XAxis
% hs{i}.XAxis. = ax_old.XAxis;
% hs{i}.YAxis = ax_old.YAxis;

%% covariance distribution
i=i+1;
hs{i} = my_subplot(4,2,i,[.65 .4]);
hs{i}.Position(1) = hs{i}.Position(1);
hold all;
h = correlation_coefficient(FR);
set(h,'MarkerSize',3);
set(gca,'Yscale','log');
xlim([-1,5]);
xlabel('');
text(0.5,-0.2,'cov','HorizontalAlignment','center');
set(gca,'fontsize',7);
%% explained variance
i=i+1;
hs{i} = my_subplot(4,2,i,[.65 .5]);
hs{i}.Position(2)=hs{i}.Position(2)+0.1;
hs{i}.Position(1) = hs{i}.Position(1)+0.5;
hold all;
n_dim = 500;
Exp_var = zeros(4,n_dim);
ii = 1;
% figure;
% hold on;
for isubject = [{'201106'},{'201116'},{'201117'},{'201125'},{'201126'},{'210106'}]
    dir_fish = getFishPath('onedrive',isubject{1});
    load(fullfile(dir_fish,'pca_result'),'explained');
    plot(1:n_dim, cumsum(explained(1:n_dim)), '-');
    Exp_var(ii,:) = cumsum(explained(1:n_dim));
    ii=ii+1;
end
hleg = legend('fish1','fish2','fish3','fish4','fish5','fish6');
% plot(mean(Exp_var),':','Color',[24, 158, 60]/255,'LineWidth',2);
% hleg = legend('fish1','fish2','fish3','fish4','mean');
% hleg = resizeLegend(hs{i});
% hleg = legend('fish1','fish2','fish3','fish4','mean');
set(hleg,'position', [hs{i}.Position(1)+0.27,hs{i}.Position(2)+0.01,0.1,0.1]);
set(hleg, 'ItemTokenSize', [4, 1],'Color','none','Box','off')
% set(gca,'FontSize',15);
% h = gca;
% h.YTickLabel = strcat(h.YTickLabel, '%');
text(0.5,-0.2,'rank','HorizontalAlignment','center');
ht=text(-.23,0.5,{'% variance','explained'},'HorizontalAlignment','center','VerticalAlignment','middle');
set(ht,'rotation',90);
grid on;
grid minor;
grid minor;
xlim([0, 500]);
set(gca,'xtick',[0 250 500]);
set(gca,'ytick',[0 250 500 750]/10);
set(gca,'fontsize',7);


%% firing rate
i=i+1;
hs{i} = my_subplot(5,1,i,[0.8 .7]);
hs{i}.Position(2)=hs{i}.Position(2)+0.25;
hs{i}.Position(1) = hs{i}.Position(1);
[~,id_sort] = sort(U(:,1),'descend');
cgray=colormap('gray');
imagesc(FR(id_sort,1:1800), [0,0.3]);
hold all;
% set(gca, 'xdata', [1 nCols]*2, 'ydata', [1 nRows]*2);
% colormap(hot);
colormap(flipud(cgray));
% colorbar;
% set(gca,'fontsize',20);
% set(gca,'xtick',[1200,2400,3600,4800,6000],'xticklabel',[2:2:10]);
% set(gca,'xtick',[600:600:3000],'xticklabel',[1:5]);
set(gca,'XTickLabel','');
set(gca,'XTick',[]);
text(0,-.05,'10 s','fontangle','normal','fontsize',7);
ylabel('neuron #');
% text(0.5,-0.05,'time (min)','HorizontalAlignment','center');
text(0.5,-0.05,'time','HorizontalAlignment','center');
set(gca,'fontsize',7);
axis tight;
line([0 100],nC*[1 1]+1,'Color','k','LineWidth',2);

%%
FR = double(FR); % firing rate

K = 1024;
rng('default');
i_rand = randsample(size(FR,1),K);
FR = FR(i_rand,:);
C = cov(double(FR'));
s = diag(C);
s= mean(s);
C = C/s;

lam = eig(C);
lam = sort(lam,'descend');

xlim_cutoff = find(cumsum(lam/sum(lam)) > 0.9);
xlim_cutoff = xlim_cutoff(1);

U_rand = normrnd(0,1/K,[K,K]);
[U_rand,~] = qr(U_rand);
C_construct = U_rand*diag(lam)*U_rand';

save("fig2_C_data.mat", "C", "C_construct", "xlim_cutoff")
% normalize diagonal %
% s = diag(C_construct);
% s=sqrt(s);
% C_construct = diag(1./s)*C_construct*diag(1./s);

%% subsampling
i=i+1;
hs{i} = my_subplot(2,2,i,[.7 .5]);
hs{i}.Position(2)=hs{i}.Position(2)+0.5;
hs{i}.Position(1) = hs{i}.Position(1);
hold all;
[h1,h2,axes1,~] = subsampling_pretty(C, true);
copyobj(axes1.Children,hs{i});
close(h1);
close(h2);
set(gca,'Xscale','log','Yscale','log');
xlim([0 xlim_cutoff/K]);
h = get(gca,'children');
% for ii = 1:length(h)
%     if strcmp(get(h(ii),'type'),'scatter') 
%         set(h(ii),'SizeData',2);
%     elseif strcmp(get(h(ii),'type'),'line')
%         set(h(ii),'MarkerSize',1.2);
%         set(h(ii),'LineWidth',0.5);
%     end
% end
hleg = legend();
set(hleg, 'ItemTokenSize', [7, 2],'Color','none')
leg_pos = get(hleg,'position') ;
set(hleg,'position',[leg_pos(1)+0.05,leg_pos(2),leg_pos(3)*0.8,leg_pos(4)]);
text(0.5,-0.1,'rank/N','HorizontalAlignment','center');
ht=text(-.2,0.5,'eigenvalue \lambda','HorizontalAlignment','center','VerticalAlignment','middle');

set(ht,'rotation',90);
set(gca,'fontsize',7);
text(0.5,1.1,'Data Cov','HorizontalAlignment','center','fontsize',10);

%%
i=i+1;
hs{i} = my_subplot(2,2,i,[.7 .5]);
hs{i}.Position(2)=hs{i}.Position(2)+0.5;
hs{i}.Position(1) = hs{i}.Position(1);
hold all;
[h1,h2,axes1,~] = subsampling_pretty(C_construct, false);
copyobj(axes1.Children,hs{i});
close(h1);
close(h2);
set(gca,'Xscale','log','Yscale','log');
xlim([0 xlim_cutoff/K]);
h = get(gca,'children');
% for ii = 1:length(h)
%     if strcmp(get(h(ii),'type'),'scatter') 
%         set(h(ii),'SizeData',2);
%     elseif strcmp(get(h(ii),'type'),'line')
%         set(h(ii),'MarkerSize',1.2);
%         set(h(ii),'LineWidth',0.5);
%     end
% end
hleg = legend();
set(hleg, 'ItemTokenSize', [7, 2],'Color','none')
leg_pos = get(hleg,'position') ;
set(hleg,'position',[leg_pos(1)+0.05,leg_pos(2),leg_pos(3)*0.8,leg_pos(4)]) ;
text(0.5,-0.1,'rank/N','HorizontalAlignment','center');
ht=text(-.2,0.5,'eigenvalue \lambda','HorizontalAlignment','center','VerticalAlignment','middle');
set(ht,'rotation',90);
set(gca,'fontsize',7);
text(0.5,1.1,'Rand Eigvec Cov','HorizontalAlignment','center','fontsize',10);





% -------------- LETTERS: A,B,C...
hp=.065; % position relative to panel
hy=1.15;
deffont=8;
for j = [1:length(hs)]
	if j==1
        hp0=.02;
		hy0=hy-0.04;
    elseif j == 2 || j ==3
        hp0 = .09;
        hy0=hy;
    elseif j == 4
        hp0 = .1;
        hy0=hy;
    else
        hp0=hp;
        hy0 = hy;
    end
    hpos = hs{j}.Position;
    axes('position', [hpos(1)-hp0 hpos(2)+hpos(4)*hy0 .01 .01]);
    if j <= 4
        text(0,0, char(64+j+1),'fontsize',10,'fontweight','bold','fontangle','normal');
    else
        text(0,0, char(64+j+2),'fontsize',10,'fontweight','bold','fontangle','normal');
    end
    axis([0 1 0 1]);
    axis off;
end
% 
% tl=squeeze(evtlag(128,:,:));
% 
% tli = interp1(tdelay,tl,[-8:.01:8]);
% 
% [~,ix1]=min(abs(max(tli,[],1)/2 - tli(1:800,:)));
% [~,ix2]=min(abs(max(tli,[],1)/2 - tli(801:end,:)));
% 
% fwhm = (800-ix1+ix2)*.01

%%
print(fullfile(matroot,'Fig_experimental_observation.pdf'),'-dpdf');
% print(fullfile(matroot,'fig2new.pdf'),'-dpdf');
