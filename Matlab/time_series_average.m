function [tsadata, navgs,rpm]=tsa_intp(data,tach,sr,ratio,navgs)

%test function definition
    dt=1/sr;
    %time bewteen samples.
    n = length(tach);
    tach = tach - mean(tach);
    x=find(tach<=0); % find tdt=1/sr;
    %time bewteen samples.
    n = length(tach);
    tach = tach - mean(tach);
    x=find(tach<=0); % find the zero crossings,
    %eliminate the first value to insure that
    i=x(find(tach(x(2:length(x))-1)>0)); %(i-1)>=1 {i.e. x(2:length(x)}
    if i(end) == n,
    i(end) = [];
    end
    i1 = i + 1;
    % now interpolate the zero crossing times
    in = i+tach(i)./(tach(i)-tach(i1));
    zct = in*dt';
    rpm = mean(1./diff(zct))*60*ratio;
    % Define the number of averages to perform
    if nargin < 6,
    navgs = floor((length(zct)-1)*ratio);
    end
    % Determine radix 2 number where # of points in resampled TSA
    % is at sample rate just greater than fsample
    N=(2^(ceil(log2(60/rpm*sr))));
    % now calculate times for each rev (1/ratio teeth pass by)
    xidx = 1:length(zct);
    % resample vibe data using zero crossing times to interpolate the vibe
    yy = zeros(1,N); %data to accumulate the resampled signal once per rev
    ya = yy;
    %ya is the resample signal once per rev
    iN = 1/N;
    %resample N points per rev
    tidx = 1;
    %start of zct index
    ir = 1/ratio; %inverse ratio - how much to advance zct
    zct1 = zct(tidx);%start zct time;
    x = (0:length(data)-1)*dt;%time index of each sample
    z = zeros(navgs,1);
    for k = 1:navgs
    tidx = tidx + ir;
    %get the zct for the shaft
    stidx = floor(tidx)-1; %start idx for interpolation
    zcti = polint(xidx,zct, stidx, 2, tidx); %interpolated ZCT
    dtrev = zcti - zct1; %time of 1 rev
    dtic = dtrev*iN;
    %time between each sample
    for j = 1:N,
    cidx = floor(zct1*sr);
    ya(j) = polint(x,data,cidx,2,zct1); %interp. time domain sample
    zct1 = zct1 + dtic; %increment to the next sample
    end
    zct1 = zcti;
    % generate resampled vibe data and accumulate t
    % the vector values for each rev
    yy = yy + ya;
    end
    tsadata = yy/navgs; % compute the average