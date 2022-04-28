clear
close all
x = -1.5:1e-4:1.5;
Lrelu = leakyrelu(x);
Tan = tanh(x);
figure
plot(x,Lrelu,'r',x,Tan,'b')

function outputs = leakyrelu(x)
alpha = 0.01;
outputs = x.*double(x<0).*alpha + x.*double(x>=0);
end