% The accuracy vs PNR plot
% for SNR -10, 0, 10 dB

clear all
close all
clc


load('minusten')
load('zero')
load('ten')


%levels = [0.001:0.0001:0.4]
levels = [0.0001:0.0001:0.0009,0.001:0.001:0.009,0.01:0.01:0.09,0.1:0.1:1];
levelsdb = 10 * log10(levels);
%levels = [0.001:0.02:0.4]
acc_zero = zeros(length(levels),1);
acc_ten = zeros(length(levels),1);
acc_minus_ten = zeros(length(levels),1);
for i = 1:length(levels)
    acc_minus_ten(i,1) = 1 - (( sum(minusten <= (10 * levels(i))) )/(length(minusten)));
    acc_zero(i,1) = 1 - ((sum(zero <= levels(i)))/(length(zero)));
    acc_ten(i,1) = 1 - ((sum(ten <= 0.1 * levels(i)))/(length(ten)));
end





figure
semilogx(levels, 0.757 * ones(length(levels),1),'b--','LineWidth',2,'MarkerSize',5)
hold on
semilogx(levels,acc_ten,'b-o','LineWidth',2,'MarkerSize',5)
hold on
semilogx(levels, 0.71 * ones(length(levels),1),'k--','LineWidth',2,'MarkerSize',5)
hold on
semilogx(levels,acc_zero,'k->','LineWidth',2,'MarkerSize',5)
hold on
semilogx(levels, 0.23 * ones(length(levels),1),'r--','LineWidth',2,'MarkerSize',5)
hold on
semilogx(levels,acc_minus_ten,'r-+','LineWidth',2,'MarkerSize',5)
xlabel('PNR [dB]')
ylabel('Accuracy %')
set(gca,'fontsize',12)
box on
grid on
lgd = legend('No attack-SNR=10 dB','SNR = 10 dB','No attack-SNR=0 dB', 'SNR = 0 dB', 'No attack-SNR=-10 dB','SNR = -10 dB')
lgd.FontSize = 12
New_XTickLabel = get(gca,'xtick');
set(gca,'XTickLabel',10*log10(New_XTickLabel));
