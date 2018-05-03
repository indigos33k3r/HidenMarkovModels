function [prob,lambda,LL] = em_poisson(data,number_comp)
% Функция [prob,lambda] = em(data,number_comp)
% реализует EM-алгоритм для смеси распределений Пуассона
% где:
%   [Параметры]
%       data - анализируемые данные
%       number_comp - число компонентов, составляющих смесь
%   [Результат]
%       prob - вероятности компонентов смеси (веса компонент)
%       lambda - интенсивности отдельных компонент
%       LL - логарифм правдоподобия

% Определим размер данных
data_size = length(data);
% Минимальное значение выборки
min_data = min(data);
% Максимальное значение выборки
max_data = max(data);
% Введем обозначение g(i,j) - неизвестная апостериорная вероятность того,
% что обучающий объект x_i получен из j-ой компоненты смеси. g(i,j)
% являются скрытыми переменными. g - матрица размером i x j, где j - число
% столбцов (число компонентов смеси), а i - число строк (обучающих элементов)
g = zeros(data_size,number_comp);
% Определим начальные значения вероятностей компонентов (веса компонент)
prob_old = [ones(number_comp,1)/number_comp];
% Определим начальные значения интенсивностей компонент
k = (max_data-min_data)/(number_comp + 1);
lambda_old = [min_data+k:k:min_data+k*number_comp];
sum_g = 0;
prob_new = 0;
lambda_new = 0;
delta = 0;
delta_stop = 4e-03; % 1e-010;
criterion_delta = 1;
%counter = 0;
% Пока скрытые переменные g(i,j) не перестанут существенно изменятся 
% (delta > 1e-010)
while criterion_delta > delta_stop
    %----------------------------------------------------------------------
    %                        E-шаг алгоритма
    %----------------------------------------------------------------------
    delta_max = 0;
    % Вычислим значения g по формуле Байеса
    for i=1:data_size
        % Вычислим полную вероятность (знаменатель формулы Байеса)g
        prob_full = 0;
        for j=1:number_comp
            prob_full = prob_full + prob_old(j)*poisspdf(data(i),lambda_old(j));
        end
        % Вычислим скрытые переменные (условные вероятности по формуле Байеса)
        for j = 1:number_comp
            g_old = g(i,j);
            g(i,j) = (prob_old(j)*poisspdf(data(i),lambda_old(j)))/prob_full;
            % Пересчитаем значение порога и критерий остановки
            delta = abs(g(i,j) - g_old);
            if delta > delta_max
                delta_max = delta;
                criterion_delta = delta_max;
            end
        end
    end
    %----------------------------------------------------------------------
    %                        M-шаг алгоритма
    %----------------------------------------------------------------------
    % Вычислим новые значения весов компонент как среднее арифметическое
    for j = 1:number_comp
        for i = 1:data_size
            sum_g = sum_g + g(i,j);
            lambda_new = lambda_new + g(i,j)*data(i);
        end
        prob_new = sum_g/data_size;
        lambda_new = lambda_new/sum_g;
        % Перепишем старый вес компонента
        prob_old(j) = prob_new;
        % Перепишем интенсивность компонента
        lambda_old(j) = lambda_new;
        sum_g = 0;
        prob_new = 0;
        lambda_new = 0;
    end
    %counter = counter + 1;
end 
prob = prob_old;
lambda = lambda_old;
% Получим логарифм правдоподобия
Lik = 0;
for i = 1:data_size
    sum_comp = 0;
    for j = 1:number_comp 
        sum_comp = sum_comp + (prob(j)*poisspdf(data(i),lambda(j)));
    end
    Lik = Lik+log(sum_comp);
end
    LL = Lik;%log(Lik);
