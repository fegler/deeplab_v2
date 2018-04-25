function net = deeplab_zoo(modelName)
%DEEPLAB_ZOO - load segmentation network by name
%  DEEPLAB_ZOO(MODELNAME) - loads a segmenter by its given name. 
%  If it cannot be found on disk, it will be downloaded via the world
%  wide web.
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  modelNames = {
    'deeplab-vggvd-t-v2' ...
    'deeplab-vggvd-v2' ...
    'deeplab-res101-t-v2' ...
    'deeplab-res101-v2' ...
  } ;

  msg = sprintf('%s: unrecognised model', modelName) ;
  assert(ismember(modelName, modelNames), msg) ;
  modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  modelPath = fullfile(modelDir, sprintf('%s.mat', modelName)) ;
  if ~exist(modelPath, 'file'), fetchModel(modelName, modelPath) ; end
  net = dagnn.DagNN.loadobj(load(modelPath)) ;

% ---------------------------------------
function fetchModel(modelName, modelPath)
% ---------------------------------------

  waiting = true ;
  prompt = sprintf(strcat('%s was not found at %s\nWould you like to ', ...
          ' download it from THE INTERNET (y/n)?\n'), modelName, modelPath) ;

  while waiting
    str = input(prompt,'s') ;
    switch str
      case 'y'
        if ~exist(fileparts(modelPath), 'dir'), mkdir(fileparts(modelPath)) ; end
        fprintf(sprintf('Downloading %s ... \n', modelName)) ;
        baseUrl = 'http://www.robots.ox.ac.uk/~albanie/models/deeplab' ;
        url = sprintf('%s/%s.mat', baseUrl, modelName) ;
        urlwrite(url, modelPath) ;
        return ;
      case 'n', throw(exception) ;
      otherwise, fprintf('input %s not recognised, please use `y/n`\n', str) ;
    end
  end
