import json
import csv
import math
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

# ===================== 1. МОДЕЛИ ДАННЫХ =====================

class InputSource(Enum):
    FILE = "file"
    MANUAL = "manual"
    API = "api"

@dataclass
class Indicator:
    id: str
    value: float
    weight: Optional[float] = 1.0
    unit: Optional[str] = ""
    region: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.weight is None:
            self.weight = 1.0

@dataclass
class CalculationInput:
    indicators: List[Indicator]
    coefficients: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalculationResult:
    success: bool
    data: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# ===================== 2. ИСКЛЮЧЕНИЯ =====================

class MetricsCalculatorError(Exception):
    pass

class ValidationError(MetricsCalculatorError):
    pass

class CalculationError(MetricsCalculatorError):
    pass

class FileReadError(MetricsCalculatorError):
    pass

class ZeroDivisionError(MetricsCalculatorError):
    pass

# ===================== 3. ВАЛИДАТОР =====================

class DataValidator:
    @staticmethod
    def validate_indicator(indicator: Indicator) -> List[str]:
        errors = []
        if not indicator.id or not isinstance(indicator.id, str):
            errors.append(f"Некорректный ID: {indicator.id}")
        if not isinstance(indicator.value, (int, float)):
            errors.append(f"Некорректное значение {indicator.id}: {indicator.value}")
        elif math.isnan(indicator.value):
            errors.append(f"Значение {indicator.id} не является числом")
        if indicator.weight is not None and indicator.weight < 0:
            errors.append(f"Вес {indicator.id} не может быть отрицательным")
        return errors
    
    @staticmethod
    def validate_input_data(data: CalculationInput) -> None:
        if not data.indicators:
            raise ValidationError("Список показателей пуст")
        all_errors = []
        for indicator in data.indicators:
            errors = DataValidator.validate_indicator(indicator)
            all_errors.extend(errors)
        ids = [ind.id for ind in data.indicators]
        if len(ids) != len(set(ids)):
            all_errors.append("ID показателей должны быть уникальными")
        for key, value in data.coefficients.items():
            if not isinstance(value, (int, float)):
                all_errors.append(f"Коэффициент {key} должен быть числом")
        if all_errors:
            raise ValidationError("Ошибки валидации:\n" + "\n".join(all_errors))

# ===================== 4. МОДУЛЬ РАСЧЁТА =====================

class MetricsCalculator:
    @staticmethod
    def calculate_arithmetic_mean(indicators: List[Indicator]) -> float:
        if not indicators:
            raise CalculationError("Нет данных для расчёта")
        values = [ind.value for ind in indicators]
        return sum(values) / len(values)
    
    @staticmethod
    def calculate_weighted_mean(indicators: List[Indicator]) -> float:
        if not indicators:
            raise CalculationError("Нет данных для расчёта")
        weighted_sum = 0
        total_weight = 0
        for ind in indicators:
            weight = ind.weight if ind.weight is not None else 1.0
            weighted_sum += ind.value * weight
            total_weight += weight
        if total_weight == 0:
            raise ZeroDivisionError("Сумма весов равна нулю")
        return weighted_sum / total_weight
    
    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        if previous == 0:
            raise ZeroDivisionError("Предыдущее значение не может быть 0")
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def calculate_efficiency_ratio(output: float, input_val: float) -> float:
        if input_val == 0:
            raise ZeroDivisionError("Входное значение не может быть 0")
        return output / input_val
    
    @staticmethod
    def calculate_standard_deviation(indicators: List[Indicator]) -> float:
        if len(indicators) < 2:
            raise CalculationError("Нужно минимум 2 значения")
        values = [ind.value for ind in indicators]
        mean = sum(values) / len(values)
        squared_differences = [(x - mean) ** 2 for x in values]
        variance = sum(squared_differences) / len(values)
        return math.sqrt(variance)
    
    @staticmethod
    def calculate_median(indicators: List[Indicator]) -> float:
        if not indicators:
            raise CalculationError("Нет данных для расчёта")
        values = sorted([ind.value for ind in indicators])
        n = len(values)
        if n % 2 == 1:
            return values[n // 2]
        else:
            return (values[n // 2 - 1] + values[n // 2]) / 2
    
    @staticmethod
    def calculate_variance(indicators: List[Indicator]) -> float:
        if len(indicators) < 2:
            raise CalculationError("Нужно минимум 2 значения")
        values = [ind.value for ind in indicators]
        mean = sum(values) / len(values)
        squared_differences = [(x - mean) ** 2 for x in values]
        return sum(squared_differences) / len(values)
    
    @staticmethod
    def calculate_range(indicators: List[Indicator]) -> Dict[str, float]:
        if not indicators:
            raise CalculationError("Нет данных для расчёта")
        values = [ind.value for ind in indicators]
        return {
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }
    
    @staticmethod
    def calculate_all_metrics(input_data: CalculationInput) -> CalculationResult:
        try:
            DataValidator.validate_input_data(input_data)
            results = {}
            errors = []
            warnings = []
            try:
                results['Среднее арифметическое'] = MetricsCalculator.calculate_arithmetic_mean(input_data.indicators)
            except CalculationError as e:
                errors.append(f"Среднее арифметическое: {e}")
            try:
                results['Средневзвешенное'] = MetricsCalculator.calculate_weighted_mean(input_data.indicators)
            except CalculationError as e:
                errors.append(f"Средневзвешенное: {e}")
            try:
                results['Стандартное отклонение'] = MetricsCalculator.calculate_standard_deviation(input_data.indicators)
            except CalculationError as e:
                warnings.append(f"Стандартное отклонение: {e}")
            try:
                results['Медиана'] = MetricsCalculator.calculate_median(input_data.indicators)
            except CalculationError as e:
                warnings.append(f"Медиана: {e}")
            try:
                results['Дисперсия'] = MetricsCalculator.calculate_variance(input_data.indicators)
            except CalculationError as e:
                warnings.append(f"Дисперсия: {e}")
            try:
                range_vals = MetricsCalculator.calculate_range(input_data.indicators)
                results.update({
                    'Минимальное значение': range_vals['min'],
                    'Максимальное значение': range_vals['max'],
                    'Размах': range_vals['range']
                })
            except CalculationError as e:
                warnings.append(f"Размах: {e}")
            if 'previous_period_value' in input_data.coefficients:
                try:
                    current_mean = results.get('Среднее арифметическое')
                    if current_mean is not None:
                        previous = input_data.coefficients['previous_period_value']
                        results['Темп роста, %'] = MetricsCalculator.calculate_growth_rate(current_mean, previous)
                except CalculationError as e:
                    errors.append(f"Темп роста: {e}")
            if ('output_value' in input_data.coefficients and 'input_value' in input_data.coefficients):
                try:
                    output = input_data.coefficients['output_value']
                    input_val = input_data.coefficients['input_value']
                    results['Коэффициент эффективности'] = MetricsCalculator.calculate_efficiency_ratio(output, input_val)
                except CalculationError as e:
                    errors.append(f"Коэффициент эффективности: {e}")
            if len(input_data.indicators) > 0:
                values = [ind.value for ind in input_data.indicators]
                results['Сумма'] = sum(values)
                results['Количество показателей'] = len(values)
                if results.get('Сумма') != 0:
                    results['Доля каждого показателя, %'] = 100 / len(values)
            success = len(errors) == 0
            return CalculationResult(success=success, data=results, errors=errors, warnings=warnings)
        except ValidationError as e:
            return CalculationResult(success=False, data={}, errors=[f"Ошибка валидации: {str(e)}"], warnings=[])
        except Exception as e:
            return CalculationResult(success=False, data={}, errors=[f"Непредвиденная ошибка: {str(e)}"], warnings=[])

# ===================== 5. ЧТЕНИЕ ИЗ ФАЙЛА =====================

class FileReader:
    @staticmethod
    def read_json(filepath: Union[str, Path]) -> CalculationInput:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            indicators = []
            for item in data.get('indicators', []):
                indicator = Indicator(
                    id=item.get('id', ''),
                    value=float(item['value']),
                    weight=float(item.get('weight', 1.0)) if item.get('weight') else 1.0,
                    unit=item.get('unit', ''),
                    region=item.get('region'),
                    timestamp=item.get('timestamp')
                )
                indicators.append(indicator)
            coefficients = data.get('coefficients', {})
            metadata = data.get('metadata', {})
            return CalculationInput(indicators=indicators, coefficients=coefficients, metadata=metadata)
        except json.JSONDecodeError as e:
            raise FileReadError(f"Ошибка JSON: {e}")
        except Exception as e:
            raise FileReadError(f"Ошибка чтения файла: {e}")
    
    @staticmethod
    def read_csv(filepath: Union[str, Path]) -> CalculationInput:
        try:
            indicators = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        indicator = Indicator(
                            id=row.get('id', f'row_{row_num}'),
                            value=float(row.get('value', 0)),
                            weight=float(row.get('weight', 1.0)) if row.get('weight') else 1.0,
                            unit=row.get('unit', ''),
                            region=row.get('region')
                        )
                        indicators.append(indicator)
                    except ValueError as e:
                        raise FileReadError(f"Ошибка в строке {row_num}: {e}")
            return CalculationInput(indicators=indicators)
        except Exception as e:
            raise FileReadError(f"Ошибка чтения CSV: {e}")
    
    @staticmethod
    def read_file(filepath: Union[str, Path]) -> CalculationInput:
        path = Path(filepath)
        if path.suffix.lower() == '.csv':
            return FileReader.read_csv(filepath)
        else:
            raise FileReadError(f"Неподдерживаемый формат файла: {path.suffix}")

# ===================== 6. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (Обновленный, современный стиль) =====================

class MetricsCalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Модуль расчёта показателей")
        self.root.geometry("1400x900")
        self.setup_styles()
        self.current_data = None
        self.calculation_history = []
        self.last_loaded_file = None
        self.last_result = None
        self.current_figure = None
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        # Современные цвета
        style.configure('.', background='#2E2E2E', foreground='white')
        style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Helvetica', 11))
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('TButton', font=('Helvetica', 11, 'bold'), background='#4CAF50', foreground='white')
        style.map('TButton', background=[('active', '#45a049')])
        style.configure('Treeview', background='#3E3E3E', foreground='white', fieldbackground='#3E3E3E', font=('Helvetica', 10))
        style.configure('Treeview.Heading', background='#555', foreground='white', font=('Helvetica', 11, 'bold'))
        style.configure('Vertical.TScrollbar', background='#555')
        self.root.configure(bg='#2E2E2E')

    def create_widgets(self):
        # Главное окно с вкладками
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка Ввод и расчет
        self.tab_input = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_input, text="Ввод и расчет")
        self.create_input_tab(self.tab_input)

        # Вкладка Результаты
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="Результаты")
        self.create_results_tab(self.tab_results)

        # Статусбар
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(self.status_frame, text="Готов к работе", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        self.data_info_label = ttk.Label(self.status_frame, text="Показателей: 0")
        self.data_info_label.pack(side=tk.RIGHT, padx=5)

    def create_input_tab(self, parent):
        # Верхний раздел: Ввод данных и коэффициенты
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая часть: Ввод данных
        data_frame = ttk.Frame(top_frame)
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(data_frame, text="Ввод данных показателей", style='Title.TLabel').pack(pady=5)
        self.create_data_input_area(data_frame)

        # Правая часть: Коэффициенты и действия
        coeff_frame = ttk.Frame(top_frame)
        coeff_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        ttk.Label(coeff_frame, text="Коэффициенты", style='Title.TLabel').pack(pady=5)
        self.create_coefficients_area(coeff_frame)

        # Нижняя часть: Кнопки
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.create_action_buttons(btn_frame)

    def create_data_input_area(self, parent):
        # Таблица для ручного ввода
        columns = ['ID', 'Значение', 'Вес', 'Ед.изм']
        self.data_tree = ttk.Treeview(parent, columns=columns, show='headings', height=12)
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120, minwidth=80)
        self.data_tree.pack(fill=tk.BOTH, expand=True)

        # Скроллы
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Ввод новых данных
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)

        self.id_entry = ttk.Entry(input_frame, width=15)
        self.id_entry.pack(side=tk.LEFT, padx=2)
        self._set_placeholder_color(self.id_entry, "ID")

        self.value_entry = ttk.Entry(input_frame, width=15)
        self.value_entry.pack(side=tk.LEFT, padx=2)
        self._set_placeholder_color(self.value_entry, "Значение")

        self.weight_entry = ttk.Entry(input_frame, width=10)
        self.weight_entry.pack(side=tk.LEFT, padx=2)
        self._set_placeholder_color(self.weight_entry, "1.0")

        self.unit_entry = ttk.Entry(input_frame, width=10)
        self.unit_entry.pack(side=tk.LEFT, padx=2)
        self._set_placeholder_color(self.unit_entry, "Ед.изм")

        # Кнопки
        btns_frame = ttk.Frame(parent)
        btns_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btns_frame, text="Добавить", command=self.add_manual_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns_frame, text="Удалить выбранное", command=self.delete_selected_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns_frame, text="Очистить всё", command=self.clear_all_records).pack(side=tk.LEFT, padx=5)
        # Удалена кнопка "Пример данных"

    def create_coefficients_area(self, parent):
        # Ввод коэффициентов
        self.prev_value_var = tk.StringVar()
        self.output_value_var = tk.StringVar()
        self.input_value_var = tk.StringVar()

        ttk.Label(parent, text="Дополнительные коэффициенты (ключ=значение):").pack(anchor=tk.W, pady=5)
        self.coeff_text = scrolledtext.ScrolledText(parent, height=6)
        self.coeff_text.pack(fill=tk.BOTH, expand=True)
        self.coeff_text.insert('1.0', 'custom_coefficent=1.0\nmultiplier=1.0\n\nФормат: ключ=значение')

    def create_action_buttons(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Расчитать", command=self.calculate_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сохранить", command=self.save_data).pack(side=tk.LEFT, padx=5)
        # Удалена кнопка "История"

    def create_results_tab(self, parent):
        # Основной текст для вывода результатов
        self.results_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=('Helvetica', 11), bg='#1E1E1E', fg='white')
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Кнопки для копирования и очистки
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Копировать", command=self.copy_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Очистить", command=self.clear_results).pack(side=tk.LEFT, padx=5)

    # ===================== Методы работы с данными =====================

    def get_current_data(self) -> CalculationInput:
        indicators = []
        for item in self.data_tree.get_children():
            values = self.data_tree.item(item)['values']
            indicator = Indicator(
                id=values[0],
                value=float(values[1]),
                weight=float(values[2]) if values[2] else 1.0,
                unit=values[3]
            )
            indicators.append(indicator)
        coefficients = {}
        try:
            if self.prev_value_var.get():
                coefficients['previous_period_value'] = float(self.prev_value_var.get())
        except:
            pass
        try:
            if self.output_value_var.get():
                coefficients['output_value'] = float(self.output_value_var.get())
        except:
            pass
        try:
            if self.input_value_var.get():
                coefficients['input_value'] = float(self.input_value_var.get())
        except:
            pass
        # Дополнительные коэффициенты
        coeff_text = self.coeff_text.get('1.0', tk.END)
        for line in coeff_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    try:
                        coefficients[key.strip()] = float(value.strip())
                    except:
                        pass
        return CalculationInput(indicators=indicators, coefficients=coefficients)

    def update_data_info(self):
        count = len(self.data_tree.get_children())
        self.data_info_label.config(text=f"Показателей: {count}")
        return count

    def add_manual_record(self):
        try:
            id_val = self.id_entry.get().strip()
            value_val = float(self.value_entry.get())
            weight_val = float(self.weight_entry.get())
            unit_val = self.unit_entry.get().strip()
            if not id_val:
                messagebox.showwarning("Внимание", "Поле ID не может быть пустым")
                return
            self.data_tree.insert('', tk.END, values=(id_val, f"{value_val:.4f}", f"{weight_val:.4f}", unit_val))
            self.id_entry.delete(0, tk.END)
            self.value_entry.delete(0, tk.END)
            self.weight_entry.delete(0, tk.END)
            self.weight_entry.insert(0, "1.0")
            self.unit_entry.delete(0, tk.END)
            self.update_data_info()
            self.status_label.config(text="Запись добавлена")
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные числовые значения")

    def delete_selected_record(self):
        selection = self.data_tree.selection()
        if selection:
            for item in selection:
                self.data_tree.delete(item)
            self.update_data_info()
            self.status_label.config(text="Запись удалена")
        else:
            messagebox.showinfo("Информация", "Выберите запись для удаления")

    def clear_all_records(self):
        if messagebox.askyesno("Подтверждение", "Очистить все записи?"):
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            self.update_data_info()
            self.status_label.config(text="Все записи очищены")

    def load_example_data(self):
        # Удалена функция загрузки примера данных
        pass

    def load_from_file(self, file_type):
        try:
            filetypes = []
            if file_type == 'csv':
                filetypes = [("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
            filename = filedialog.askopenfilename(title="Выберите файл", filetypes=filetypes)
            if filename:
                self.last_loaded_file = filename
                input_data = FileReader.read_file(filename)
                self.clear_all_records()
                for indicator in input_data.indicators:
                    self.data_tree.insert('', tk.END, values=(
                        indicator.id,
                        f"{indicator.value:.4f}",
                        f"{indicator.weight:.4f}",
                        indicator.unit or ""
                    ))
                # Загрузка коэффициентов
                if 'previous_period_value' in input_data.coefficients:
                    self.prev_value_var.set(str(input_data.coefficients['previous_period_value']))
                if 'output_value' in input_data.coefficients:
                    self.output_value_var.set(str(input_data.coefficients['output_value']))
                if 'input_value' in input_data.coefficients:
                    self.input_value_var.set(str(input_data.coefficients['input_value']))
                count = self.update_data_info()
                self.status_label.config(text=f"Файл загружен: {Path(filename).name} ({count} показателей)")
                self.current_data = input_data
                if input_data.indicators:
                    self.calculate_metrics()
                else:
                    messagebox.showwarning("Внимание", "Файл не содержит данных для анализа")
        except FileReadError as e:
            messagebox.showerror("Ошибка чтения файла", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def calculate_metrics(self):
        try:
            input_data = self.get_current_data()
            if not input_data.indicators:
                messagebox.showwarning("Внимание", "Нет данных для расчёта")
                return
            self.status_label.config(text="Выполняется расчет...")
            self.root.update()
            result = MetricsCalculator.calculate_all_metrics(input_data)
            self.display_results(result)
            self.save_to_history(result, input_data)
            self.status_label.config(text="Расчет выполнен успешно")
        except Exception as e:
            messagebox.showerror("Ошибка расчета", str(e))
            self.status_label.config(text="Ошибка при расчете")

    def display_results(self, result: CalculationResult):
        self.results_text.delete('1.0', tk.END)
        if not result.success:
            self.results_text.insert('1.0', "❌ РАСЧЁТ НЕ ВЫПОЛНЕН\n\n")
            for error in result.errors:
                self.results_text.insert(tk.END, f"• {error}\n")
            return
        source_info = ""
        if self.last_loaded_file:
            source_info = f"Источник данных: {Path(self.last_loaded_file).name}\n"
        output = "Результаты анализа данных:\n\n"
        main_metrics = [
            'Среднее арифметическое',
            'Средневзвешенное',
            'Стандартное отклонение',
            'Дисперсия',
            'Медиана',
            'Минимальное значение',
            'Максимальное значение',
            'Размах',
            'Сумма',
            'Количество показателей'
        ]
        for metric in main_metrics:
            if metric in result.data:
                value = result.data[metric]
                if 'Количество' in metric:
                    output += f"{metric}: {int(value)}\n"
                else:
                    output += f"{metric}: {value:.6f}\n"
        output += "\nДополнительные рассчёты:\n"
        additional_metrics = [
            'Темп роста, %',
            'Коэффициент эффективности',
            'Доля каждого показателя, %'
        ]
        for metric in additional_metrics:
            if metric in result.data:
                value = result.data[metric]
                if 'Темп роста' in metric:
                    output += f"{metric}: {value:+.4f}%\n"
                elif 'Доля' in metric:
                    output += f"{metric}: {value:.4f}%\n"
                else:
                    output += f"{metric}: {value:.6f}\n"
        # Ошибки и предупреждения
        if result.errors:
            output += "\n⚠️ ОШИБКИ:\n"
            output += "═" * 60 + "\n"
            for error in result.errors:
                output += f"• {error}\n"
        if result.warnings:
            output += "\nℹ️ ПРЕДУПРЕЖДЕНИЯ:\n"
            output += "═" * 60 + "\n"
            for warning in result.warnings:
                output += f"• {warning}\n"
        self.last_result = result
        self.current_data = self.get_current_data()
        self.results_text.insert('1.0', output)

    def save_to_history(self, result: CalculationResult, input_data: CalculationInput):
        # История не реализована
        pass

    def update_history_display(self):
        # История не реализована
        pass

    def copy_results(self):
        text = self.results_text.get('1.0', tk.END)
        if text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Результаты скопированы в буфер обмена")

    def clear_results(self):
        if messagebox.askyesno("Подтверждение", "Очистить результаты?"):
            self.results_text.delete('1.0', tk.END)
            self.status_label.config(text="Результаты очищены")

    def save_data(self):
        # Экспорт только в CSV
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv")]
            )
            if filename:
                input_data = self.get_current_data()
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'value', 'weight', 'unit', 'region'])
                    for ind in input_data.indicators:
                        writer.writerow([ind.id, ind.value, ind.weight, ind.unit or "", ind.region or ""])
                self.status_label.config(text=f"Данные сохранены: {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def export_results(self):
        # Экспорт только в CSV
        if not hasattr(self, 'last_result'):
            messagebox.showwarning("Внимание", "Нет результатов для экспорта")
            return
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV файлы", "*.csv")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Показатель', 'Значение'])
                    for key, value in self.last_result.data.items():
                        writer.writerow([key, value])
                self.status_label.config(text=f"Результаты экспортированы: {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", str(e))

    def load_history(self):
        # История не реализована
        pass

    def _set_placeholder_color(self, entry, placeholder):
        # Установка чёрного цвета для текста-плейсхолдера
        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(foreground='black')
        def on_focus_out(event):
            if not entry.get():
                entry.insert(0, placeholder)
                entry.config(foreground='grey')
        entry.insert(0, placeholder)
        entry.config(foreground='grey')
        entry.bind('<FocusIn>', on_focus_in)
        entry.bind('<FocusOut>', on_focus_out)

# ===================== 9. ТОЧКА ВХОДА =====================

def main():
    try:
        root = tk.Tk()
        root.minsize(1200, 800)
        app = MetricsCalculatorGUI(root)
        # Центрирование окна
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        root.mainloop()
    except Exception as e:
        print(f"Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        input("Нажмите Enter для выхода...")

# ===================== 10. ТЕСТЫ =====================

def run_tests():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ")
    print("=" * 60)
    # Тестовые данные
    test_indicators = [
        Indicator(id="test1", value=100.0, weight=1.0),
        Indicator(id="test2", value=200.0, weight=2.0),
        Indicator(id="test3", value=300.0, weight=1.5),
    ]
    test_input = CalculationInput(
        indicators=test_indicators,
        coefficients={
            'previous_period_value': 150.0,
            'output_value': 600.0,
            'input_value': 200.0
        }
    )
    print("\n1. Созданы тестовые данные")
    for ind in test_indicators:
        print(f"   {ind.id}: {ind.value} (вес: {ind.weight})")
    print("\n2. Валидация данных")
    try:
        DataValidator.validate_input_data(test_input)
        print("   ✓ Валидация прошла успешно")
    except ValidationError as e:
        print(f"   ✗ Валидация не прошла: {e}")
    print("\n3. Расчет")
    result = MetricsCalculator.calculate_all_metrics(test_input)
    if result.success:
        print("   ✓ Расчет выполнен")
        for k, v in result.data.items():
            print(f"   {k}: {v}")
    else:
        print("   ✗ Есть ошибки")
        for err in result.errors:
            print(f"   {err}")
    # Тест чтения файла
    import tempfile
    test_csv_content = [
        ['id', 'value', 'weight', 'unit', 'region'],
        ['item1', '100', '1.0', 'шт.', 'Region1'],
        ['item2', '200', '2.0', 'шт.', 'Region2']
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_csv_content)
        csv_path = f.name
    try:
        loaded = FileReader.read_csv(csv_path)
        print(f"   ✓ Прочитан CSV: {len(loaded.indicators)} показателей")
        for ind in loaded.indicators:
            print(f"     {ind.id}, region: {ind.region}")
    except Exception as e:
        print(f"   ✗ Ошибка чтения CSV: {e}")
    os.remove(csv_path)
    print("=" * 60)
    print("ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)

# ===================== запуск =====================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()