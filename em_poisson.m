function [prob,lambda,LL] = em_poisson(data,number_comp)
% ������� [prob,lambda] = em(data,number_comp)
% ��������� EM-�������� ��� ����� ������������� ��������
% ���:
%   [���������]
%       data - ������������� ������
%       number_comp - ����� �����������, ������������ �����
%   [���������]
%       prob - ����������� ����������� ����� (���� ���������)
%       lambda - ������������� ��������� ���������
%       LL - �������� �������������

% ��������� ������ ������
data_size = length(data);
% ����������� �������� �������
min_data = min(data);
% ������������ �������� �������
max_data = max(data);
% ������ ����������� g(i,j) - ����������� ������������� ����������� ����,
% ��� ��������� ������ x_i ������� �� j-�� ���������� �����. g(i,j)
% �������� �������� �����������. g - ������� �������� i x j, ��� j - �����
% �������� (����� ����������� �����), � i - ����� ����� (��������� ���������)
g = zeros(data_size,number_comp);
% ��������� ��������� �������� ������������ ����������� (���� ���������)
prob_old = [ones(number_comp,1)/number_comp];
% ��������� ��������� �������� �������������� ���������
k = (max_data-min_data)/(number_comp + 1);
lambda_old = [min_data+k:k:min_data+k*number_comp];
sum_g = 0;
prob_new = 0;
lambda_new = 0;
delta = 0;
delta_stop = 4e-03; % 1e-010;
criterion_delta = 1;
%counter = 0;
% ���� ������� ���������� g(i,j) �� ���������� ����������� ��������� 
% (delta > 1e-010)
while criterion_delta > delta_stop
    %----------------------------------------------------------------------
    %                        E-��� ���������
    %----------------------------------------------------------------------
    delta_max = 0;
    % �������� �������� g �� ������� ������
    for i=1:data_size
        % �������� ������ ����������� (����������� ������� ������)g
        prob_full = 0;
        for j=1:number_comp
            prob_full = prob_full + prob_old(j)*poisspdf(data(i),lambda_old(j));
        end
        % �������� ������� ���������� (�������� ����������� �� ������� ������)
        for j = 1:number_comp
            g_old = g(i,j);
            g(i,j) = (prob_old(j)*poisspdf(data(i),lambda_old(j)))/prob_full;
            % ����������� �������� ������ � �������� ���������
            delta = abs(g(i,j) - g_old);
            if delta > delta_max
                delta_max = delta;
                criterion_delta = delta_max;
            end
        end
    end
    %----------------------------------------------------------------------
    %                        M-��� ���������
    %----------------------------------------------------------------------
    % �������� ����� �������� ����� ��������� ��� ������� ��������������
    for j = 1:number_comp
        for i = 1:data_size
            sum_g = sum_g + g(i,j);
            lambda_new = lambda_new + g(i,j)*data(i);
        end
        prob_new = sum_g/data_size;
        lambda_new = lambda_new/sum_g;
        % ��������� ������ ��� ����������
        prob_old(j) = prob_new;
        % ��������� ������������� ����������
        lambda_old(j) = lambda_new;
        sum_g = 0;
        prob_new = 0;
        lambda_new = 0;
    end
    %counter = counter + 1;
end 
prob = prob_old;
lambda = lambda_old;
% ������� �������� �������������
Lik = 0;
for i = 1:data_size
    sum_comp = 0;
    for j = 1:number_comp 
        sum_comp = sum_comp + (prob(j)*poisspdf(data(i),lambda(j)));
    end
    Lik = Lik+log(sum_comp);
end
    LL = Lik;%log(Lik);
