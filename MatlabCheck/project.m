function projection = project(signal, axis)
% axis = 1 for lateral 2 for axial
    projection = max(signal,[],axis);
    
    