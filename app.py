from flask import Flask, render_template, request, jsonify
import json
import traceback
import os
from datetime import datetime

# Import the enhanced compiler
from compiler import MiniLangCompiler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced-minilang-compiler-secret-key-2024'

# Global statistics for monitoring
compilation_stats = {
    'total_compilations': 0,
    'successful_compilations': 0,
    'optimization_runs': 0,
    'assembly_generations': 0
}

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/compile', methods=['POST'])
def compile_code():
    """Enhanced API endpoint for code compilation"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        optimize = data.get('optimize', True)
        target_arch = data.get('target_arch', 'x86_64')
        
        compilation_stats['total_compilations'] += 1
        
        if not source_code.strip():
            return jsonify({
                'success': False,
                'error': 'No source code provided'
            })
        
        # Create compiler instance and compile
        compiler = MiniLangCompiler()
        results = compiler.compile(source_code, optimize, target_arch)
        
        if results['success']:
            compilation_stats['successful_compilations'] += 1
            if optimize:
                compilation_stats['optimization_runs'] += 1
            compilation_stats['assembly_generations'] += 1
        
        # Add compilation metadata
        results['metadata'] = {
            'compilation_time': datetime.now().isoformat(),
            'compiler_version': '2.0.0',
            'optimization_enabled': optimize,
            'target_architecture': target_arch
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/examples')
def get_examples():
    """Enhanced API endpoint for example programs with comprehensive array support"""
    examples = {
        'hello_world': {
            'name': 'Hello World',
            'description': 'Simple MiniLang++ program',
            'code': '''int main() {
    return 42;  
}'''
        },
        'arrays_basic': {
            'name': 'Basic Array Operations',
            'description': 'Array declaration, initialization, and access',
            'code': '''int main() {
    int arr[5];
    int i = 0;
    
    for (i = 0; i < 5; i++) {
        arr[i] = i * 2;
    }
    
    int sum = 0;
    for (i = 0; i < 5; i++) {
        sum = sum + arr[i];
    }
    
    return sum;  
}'''
        },
        'arrays_advanced': {
            'name': 'Advanced Array Processing',
            'description': 'Array parameter passing and complex operations',
            'code': '''int findMax(int data[], int size) {
    int max = data[0];
    int i = 1;
    
    while (i < size) {
        if (data[i] > max) {
            max = data[i];
        }
        i++;
    }
    
    return max;
}

float calculateAverage(int data[], int size) {
    float sum = 0.0;
    int i = 0;
    
    for (i = 0; i < size; i++) {
        sum = sum + data[i];
    }
    
    return sum / size;
}

int main() {
    int numbers[6];
    int i = 0;
    
    numbers[0] = 1;
    numbers[1] = 1;
    for (i = 2; i < 6; i++) {
        numbers[i] = numbers[i-1] + numbers[i-2];
    }
    
    int maximum = findMax(numbers, 6);
    float average = calculateAverage(numbers, 6);
    
    if (average > 5.0 && maximum > 10) {
        return 1;  
    } else {
        return 0;  
    }
}'''
        },
        'sorting_algorithm': {
            'name': 'Bubble Sort Implementation',
            'description': 'Complete sorting algorithm with arrays',
            'code': '''void bubbleSort(int arr[], int n) {
    int i = 0;
    int j = 0;
    int temp = 0;
    bool swapped = false;
    
    for (i = 0; i < n - 1; i++) {
        swapped = false;
        
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        
        if (!swapped) {
            break;
        }
    }
}

bool isSorted(int arr[], int n) {
    int i = 0;
    for (i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

int main() {
    int data[8];
    int i = 0;
    
    for (i = 0; i < 8; i++) {
        data[i] = 8 - i;
    }
    
    bubbleSort(data, 8);
    
    if (isSorted(data, 8)) {
        return 1;  
    } else {
        return 0;  
    }
}'''
        },
        'optimization_demo': {
            'name': 'Code Optimization Showcase',
            'description': 'Code designed to demonstrate compiler optimizations',
            'code': '''int demonstrateOptimizations() {
    int x = 5 + 0;          
    int y = x * 1;          
    int z = y - 0;          
    int w = z / 1;          
    
    int unused = 999;       
    
    bool alwaysTrue = true && true;    
    bool alwaysFalse = false || false; 
    
    int redundant = w + 0;  
    
    if (alwaysTrue) {       
        return redundant;
    } else {
        return 0 * x;       
    }
}

int loopOptimization() {
    int result = 0;
    int temp = 1;           
    int i = 0;
    
    while (i < 3) {
        result = result + (i * temp);  
        i = i + 1;
    }
    
    return result;
}

int main() {
    int demo = demonstrateOptimizations();
    int loop_result = loopOptimization();
    return demo + loop_result;
}'''
        },
        'control_flow_complex': {
            'name': 'Complex Control Flow',
            'description': 'Nested loops, conditionals, and break/continue',
            'code': '''bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    int i = 5;
    while (i * i <= n) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
        i = i + 6;
    }
    
    return true;
}

int main() {
    int primes[10];
    int primeCount = 0;
    int candidate = 2;
    
    while (primeCount < 10) {
        if (isPrime(candidate)) {
            primes[primeCount] = candidate;
            primeCount++;
        }
        candidate++;
        
        if (candidate > 100) {
            break;
        }
    }
    
    int sum = 0;
    int i = 0;
    for (i = 0; i < primeCount; i++) {
        sum = sum + primes[i];
    }
    
    return sum;
}'''
        },
        'matrix_operations': {
            'name': 'Matrix Operations (2D Arrays)',
            'description': 'Simulated 2D array operations using 1D arrays',
            'code': '''
int getMatrixElement(int matrix[], int row, int col, int cols) {
    return matrix[row * cols + col];
}

void setMatrixElement(int matrix[], int row, int col, int cols, int value) {
    matrix[row * cols + col] = value;
}

void multiplyMatrices(int a[], int b[], int result[], int size) {
    int i = 0;
    int j = 0;
    int k = 0;

    for (i = 0; i < size * size; i++) {
        result[i] = 0;
    }
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k < size; k++) {
                int temp = getMatrixElement(result, i, j, size);
                int a_val = getMatrixElement(a, i, k, size);
                int b_val = getMatrixElement(b, k, j, size);
                setMatrixElement(result, i, j, size, temp + (a_val * b_val));
            }
        }
    }
}

int main() {
    int matrixA[9];  
    int matrixB[9];  
    int result[9];   
    int i = 0;
    
    for (i = 0; i < 9; i++) {
        matrixA[i] = 0;
        matrixB[i] = 0;
    }
    
    setMatrixElement(matrixA, 0, 0, 3, 1);
    setMatrixElement(matrixA, 1, 1, 3, 2);
    setMatrixElement(matrixA, 2, 2, 3, 3);
    
    setMatrixElement(matrixB, 0, 0, 3, 2);
    setMatrixElement(matrixB, 1, 1, 3, 3);
    setMatrixElement(matrixB, 2, 2, 3, 4);
    
    multiplyMatrices(matrixA, matrixB, result, 3);
    
    int trace = 0;
    trace = trace + getMatrixElement(result, 0, 0, 3);
    trace = trace + getMatrixElement(result, 1, 1, 3);
    trace = trace + getMatrixElement(result, 2, 2, 3);
    
    return trace;  
}'''
        },
        'error_examples': {
            'name': 'Common Error Examples',
            'description': 'Examples of various compilation errors',
            'code': '''int main() {
    int x = 5 @ 3;
    
    int y = 10
    
    z = 15;
    
    bool flag = 42;
    
    int arr[5];
    arr["hello"] = 10;
    
    return 0;
}'''
        }
    }
    return jsonify(examples)

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """Comprehensive code analysis endpoint"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        
        compiler = MiniLangCompiler()
        results = compiler.compile(source_code, optimize=True)
        
        # Enhanced analysis
        analysis = {
            'compilation_results': results,
            'code_quality': analyze_code_quality(results),
            'complexity_metrics': calculate_enhanced_complexity(results),
            'optimization_impact': analyze_optimization_impact(source_code),
            'suggestions': generate_enhanced_suggestions(results),
            'performance_estimate': estimate_performance(results)
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis error: {str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/api/optimize', methods=['POST'])
def optimize_code():
    """API endpoint specifically for optimization analysis"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        
        compiler = MiniLangCompiler()
        
        # Compile without optimization
        results_unoptimized = compiler.compile(source_code, optimize=False)
        if not results_unoptimized['success']:
            return jsonify(results_unoptimized)
        
        # Compile with optimization
        results_optimized = compiler.compile(source_code, optimize=True)
        
        # Detailed optimization analysis
        optimization_analysis = {
            'before': {
                'instructions': len(results_unoptimized['intermediate_code']),
                'complexity': estimate_instruction_complexity(results_unoptimized['intermediate_code'])
            },
            'after': {
                'instructions': len(results_optimized['optimized_code']),
                'complexity': estimate_instruction_complexity(results_optimized['optimized_code'])
            },
            'improvements': {
                'instruction_reduction': 0,
                'complexity_reduction': 0,
                'percentage_improvement': 0
            },
            'optimizations_applied': results_optimized['optimizations_applied'],
            'optimization_categories': categorize_optimizations(results_optimized['optimizations_applied']),
            'recommendations': generate_optimization_recommendations(results_optimized)
        }
        
        # Calculate improvements
        if optimization_analysis['before']['instructions'] > 0:
            reduction = optimization_analysis['before']['instructions'] - optimization_analysis['after']['instructions']
            optimization_analysis['improvements']['instruction_reduction'] = reduction
            optimization_analysis['improvements']['percentage_improvement'] = (reduction / optimization_analysis['before']['instructions']) * 100
        
        complexity_reduction = optimization_analysis['before']['complexity'] - optimization_analysis['after']['complexity']
        optimization_analysis['improvements']['complexity_reduction'] = complexity_reduction
        
        return jsonify({
            'success': True,
            'unoptimized': results_unoptimized,
            'optimized': results_optimized,
            'analysis': optimization_analysis
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Optimization analysis error: {str(e)}'
        })

@app.route('/api/assembly', methods=['POST'])
def generate_assembly():
    """Enhanced assembly generation endpoint"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        target_arch = data.get('target_arch', 'x86_64')
        optimize = data.get('optimize', True)
        
        compiler = MiniLangCompiler()
        results = compiler.compile(source_code, optimize, target_arch)
        
        if results['success']:
            # Enhanced assembly analysis
            assembly_lines = results['assembly_code'].split('\n')
            assembly_analysis = analyze_assembly_code(assembly_lines, target_arch)
            results['assembly_analysis'] = assembly_analysis
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Assembly generation error: {str(e)}'
        })

@app.route('/api/validate', methods=['POST'])
def validate_syntax():
    """Enhanced syntax validation with detailed feedback"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        
        compiler = MiniLangCompiler()
        
        # Lexical analysis
        lexer = LexicalAnalyzer(source_code)
        tokens = lexer.tokenize()
        
        if lexer.errors:
            return jsonify({
                'valid': False,
                'phase': 'lexical',
                'errors': [str(e) for e in lexer.errors],
                'error_count': len(lexer.errors),
                'suggestions': generate_lexical_suggestions(lexer.errors)
            })
        
        # Syntax analysis
        parser = Parser(tokens)
        ast = parser.parse()
        
        if parser.errors:
            return jsonify({
                'valid': False,
                'phase': 'syntax',
                'errors': [str(e) for e in parser.errors],
                'error_count': len(parser.errors),
                'suggestions': generate_syntax_suggestions(parser.errors)
            })
        
        return jsonify({
            'valid': True,
            'message': 'Syntax is valid',
            'token_count': len([t for t in tokens if t.type != TokenType.EOF]),
            'ast_nodes': count_ast_nodes(ast)
        })
    
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': f'Validation error: {str(e)}'
        })

@app.route('/api/format', methods=['POST'])
def format_code():
    """Enhanced code formatting endpoint"""
    try:
        data = request.get_json()
        source_code = data.get('code', '')
        style = data.get('style', 'standard')  # standard, compact, expanded
        
        formatted_code = format_minilang_code(source_code, style)
        
        return jsonify({
            'success': True,
            'formatted_code': formatted_code,
            'style_applied': style,
            'formatting_changes': count_formatting_changes(source_code, formatted_code)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Formatting error: {str(e)}'
        })

@app.route('/api/stats', methods=['GET'])
def get_compilation_stats():
    """Get compilation statistics"""
    return jsonify({
        'statistics': compilation_stats,
        'uptime': datetime.now().isoformat(),
        'version': '2.0.0'
    })

# Helper functions

def analyze_code_quality(results):
    """Analyze code quality metrics"""
    if not results['success']:
        return {'score': 0, 'issues': ['Compilation failed'], 'warnings': []}
    
    quality_score = 100
    issues = []
    warnings = []
    
    # Get warnings from compilation results
    compilation_warnings = results.get('warnings', [])
    warnings.extend(compilation_warnings)
    
    # Minor penalty for warnings (unused variables)
    quality_score -= len(compilation_warnings) * 2
    
    # Check for actual semantic errors (not warnings)
    semantic_errors = results.get('errors', {}).get('semantic', [])
    for error in semantic_errors:
        if not ('Warning:' in error or 'Unused' in error):
            issues.append(error)
            quality_score -= 10
    
    # Check function complexity
    functions = results.get('ast', {}).get('functions', [])
    for func in functions:
        complexity = calculate_function_complexity(func)
        if complexity > 10:
            quality_score -= 5
            issues.append(f"High complexity in function '{func.get('name', 'unknown')}'")
    
    return {
        'score': max(0, quality_score),
        'issues': issues,
        'warnings': warnings
    }

def calculate_enhanced_complexity(results):
    """Calculate enhanced complexity metrics"""
    if not results['success']:
        return {}
    
    ast = results.get('ast', {})
    functions = ast.get('functions', [])
    
    metrics = {
        'cyclomatic_complexity': 1,
        'cognitive_complexity': 0,
        'function_count': len(functions),
        'variable_count': 0,
        'loop_count': 0,
        'conditional_count': 0,
        'max_nesting_depth': 0,
        'lines_of_code': results.get('statistics', {}).get('lines_of_code', 0)
    }
    
    for func in functions:
        func_complexity = analyze_function_complexity_detailed(func)
        metrics['cyclomatic_complexity'] += func_complexity['cyclomatic']
        metrics['cognitive_complexity'] += func_complexity['cognitive']
        metrics['loop_count'] += func_complexity['loops']
        metrics['conditional_count'] += func_complexity['conditionals']
        metrics['max_nesting_depth'] = max(metrics['max_nesting_depth'], func_complexity['max_depth'])
    
    # Count variables from symbol table
    symbol_table = results.get('symbol_table', {})
    for scope in symbol_table.values():
        for symbol_info in scope.get('symbols', {}).values():
            if symbol_info.get('kind') in ['variable', 'parameter']:
                metrics['variable_count'] += 1
    
    return metrics

def analyze_function_complexity_detailed(function):
    """Detailed function complexity analysis"""
    complexity = {
        'cyclomatic': 0,
        'cognitive': 0,
        'loops': 0,
        'conditionals': 0,
        'max_depth': 0
    }
    
    def analyze_node(node, depth=0):
        if not isinstance(node, dict):
            return
        
        complexity['max_depth'] = max(complexity['max_depth'], depth)
        node_type = node.get('type', '')
        
        if node_type in ['IfStatement']:
            complexity['cyclomatic'] += 1
            complexity['cognitive'] += 1 + depth  # Cognitive complexity increases with nesting
            complexity['conditionals'] += 1
        elif node_type in ['WhileStatement', 'ForStatement']:
            complexity['cyclomatic'] += 1
            complexity['cognitive'] += 1 + depth
            complexity['loops'] += 1
        elif node_type in ['BreakStatement', 'ContinueStatement']:
            complexity['cognitive'] += 1
        
        # Recursively analyze child nodes
        for key, value in node.items():
            if key != 'type':
                if isinstance(value, list):
                    for item in value:
                        analyze_node(item, depth + 1 if node_type in ['IfStatement', 'WhileStatement', 'ForStatement', 'Block'] else depth)
                elif isinstance(value, dict):
                    analyze_node(value, depth + 1 if node_type in ['IfStatement', 'WhileStatement', 'ForStatement', 'Block'] else depth)
    
    if 'body' in function:
        for stmt in function['body']:
            analyze_node(stmt)
    
    return complexity

def analyze_optimization_impact(source_code):
    """Analyze potential optimization impact"""
    compiler = MiniLangCompiler()
    
    # Quick compilation without optimization
    results_no_opt = compiler.compile(source_code, optimize=False)
    if not results_no_opt['success']:
        return {'impact': 'unknown', 'reason': 'compilation_failed'}
    
    # Quick compilation with optimization
    results_opt = compiler.compile(source_code, optimize=True)
    
    original_count = len(results_no_opt['intermediate_code'])
    optimized_count = len(results_opt['optimized_code'])
    
    if original_count == 0:
        return {'impact': 'none', 'reason': 'no_code'}
    
    reduction_percentage = ((original_count - optimized_count) / original_count) * 100
    
    if reduction_percentage > 20:
        impact = 'high'
    elif reduction_percentage > 10:
        impact = 'medium'
    elif reduction_percentage > 0:
        impact = 'low'
    else:
        impact = 'none'
    
    return {
        'impact': impact,
        'reduction_percentage': reduction_percentage,
        'original_instructions': original_count,
        'optimized_instructions': optimized_count
    }

def generate_enhanced_suggestions(results):
    """Generate enhanced code improvement suggestions"""
    suggestions = []
    
    if not results['success']:
        errors = results.get('errors', {})
        
        if errors.get('lexical'):
            suggestions.append({
                'type': 'error',
                'title': 'Lexical Errors',
                'message': 'Check for invalid characters, unterminated strings, or malformed numbers.',
                'priority': 'high',
                'category': 'syntax'
            })
        
        if errors.get('syntax'):
            suggestions.append({
                'type': 'error',
                'title': 'Syntax Errors',
                'message': 'Review code structure, check for missing semicolons, brackets, or parentheses.',
                'priority': 'high',
                'category': 'syntax'
            })
        
        if errors.get('semantic'):
            suggestions.append({
                'type': 'error',
                'title': 'Semantic Errors',
                'message': 'Check variable declarations, type compatibility, and function signatures.',
                'priority': 'high',
                'category': 'semantics'
            })
    else:
        # Analyze successful compilation for improvements
        complexity = calculate_enhanced_complexity(results)
        
        if complexity['cyclomatic_complexity'] > 20:
            suggestions.append({
                'type': 'warning',
                'title': 'High Cyclomatic Complexity',
                'message': f'Complexity score: {complexity["cyclomatic_complexity"]}. Consider breaking down complex functions.',
                'priority': 'medium',
                'category': 'maintainability'
            })
        
        if complexity['max_nesting_depth'] > 4:
            suggestions.append({
                'type': 'info',
                'title': 'Deep Nesting',
                'message': f'Maximum nesting depth: {complexity["max_nesting_depth"]}. Consider extracting nested logic into functions.',
                'priority': 'low',
                'category': 'readability'
            })
        
        # Check optimization opportunities
        optimizations = results.get('optimizations_applied', [])
        if len(optimizations) > 0:
            suggestions.append({
                'type': 'success',
                'title': 'Optimization Success',
                'message': f'{len(optimizations)} optimizations applied. Your code has been improved!',
                'priority': 'info',
                'category': 'performance'
            })
    
    return suggestions

def estimate_performance(results):
    """Estimate code performance characteristics"""
    if not results['success']:
        return {'estimate': 'unknown'}
    
    intermediate_code = results.get('optimized_code', results.get('intermediate_code', []))
    
    # Simple performance estimation based on instruction types
    performance_score = 0
    instruction_weights = {
        'ASSIGN': 1,
        '+': 1, '-': 1,
        '*': 3, '/': 10, '%': 10,
        'CALL': 5,
        'ARRAY_ACCESS': 2,
        'GOTO': 1,
        'LABEL': 0
    }
    
    for instr in intermediate_code:
        op = instr.get('operation', '')
        weight = instruction_weights.get(op, 2)  # Default weight
        performance_score += weight
    
    # Categorize performance
    if performance_score < 50:
        category = 'excellent'
    elif performance_score < 100:
        category = 'good'
    elif performance_score < 200:
        category = 'fair'
    else:
        category = 'poor'
    
    return {
        'estimate': category,
        'score': performance_score,
        'instruction_count': len(intermediate_code),
        'complexity_factor': performance_score / max(1, len(intermediate_code))
    }

def estimate_instruction_complexity(instructions):
    """Estimate computational complexity of instruction sequence"""
    complexity = 0
    weights = {
        '+': 1, '-': 1, '*': 2, '/': 3, '%': 3,
        'CALL': 5, 'ARRAY_ACCESS': 2, 'ASSIGN': 1
    }
    
    for instr in instructions:
        op = instr.get('operation', '')
        complexity += weights.get(op, 1)
    
    return complexity

def categorize_optimizations(optimizations):
    """Enhanced optimization categorization"""
    categories = {
        'constant_folding': 0,
        'copy_propagation': 0,
        'dead_code_elimination': 0,
        'algebraic_simplification': 0,
        'control_flow': 0,
        'redundancy_elimination': 0
    }
    
    for opt in optimizations:
        opt_lower = opt.lower()
        if 'constant folding' in opt_lower:
            categories['constant_folding'] += 1
        elif 'copy propagation' in opt_lower:
            categories['copy_propagation'] += 1
        elif 'dead code' in opt_lower:
            categories['dead_code_elimination'] += 1
        elif 'algebraic' in opt_lower:
            categories['algebraic_simplification'] += 1
        elif 'jump' in opt_lower or 'unreachable' in opt_lower:
            categories['control_flow'] += 1
        elif 'redundant' in opt_lower:
            categories['redundancy_elimination'] += 1
    
    return categories

def generate_optimization_recommendations(results):
    """Generate specific optimization recommendations"""
    recommendations = []
    
    optimizations = results.get('optimizations_applied', [])
    optimization_categories = categorize_optimizations(optimizations)
    
    if optimization_categories['constant_folding'] > 0:
        recommendations.append("Consider using named constants for frequently used values")
    
    if optimization_categories['dead_code_elimination'] > 0:
        recommendations.append("Remove unused variables and code paths")
    
    if optimization_categories['algebraic_simplification'] > 0:
        recommendations.append("Review mathematical expressions for simplification opportunities")
    
    if len(optimizations) == 0:
        recommendations.append("Code is already well-optimized or contains complex logic")
    
    return recommendations

def analyze_assembly_code(assembly_lines, target_arch):
    """Analyze generated assembly code"""
    analysis = {
        'total_lines': len(assembly_lines),
        'data_section_lines': 0,
        'text_section_lines': 0,
        'instruction_count': 0,
        'label_count': 0,
        'jump_count': 0,
        'function_count': 0,
        'target_architecture': target_arch,
        'estimated_size_bytes': 0
    }
    
    in_data_section = False
    in_text_section = False
    
    for line in assembly_lines:
        line = line.strip()
        if not line:
            continue
        
        if '.data' in line:
            in_data_section = True
            in_text_section = False
        elif '.text' in line:
            in_data_section = False
            in_text_section = True
        elif line.endswith(':'):
            analysis['label_count'] += 1
            if line in ['main:', '_start:'] or '(' not in line:
                analysis['function_count'] += 1
        elif in_data_section:
            analysis['data_section_lines'] += 1
            if '.asciz' in line or '.string' in line:
                analysis['estimated_size_bytes'] += len(line)
        elif in_text_section and not line.startswith('.'):
            analysis['text_section_lines'] += 1
            analysis['instruction_count'] += 1
            analysis['estimated_size_bytes'] += 4  # Rough estimate
            
            if any(jmp in line for jmp in ['jmp', 'je', 'jne', 'jl', 'jg', 'call']):
                analysis['jump_count'] += 1
    
    return analysis

def generate_lexical_suggestions(errors):
    """Generate suggestions for lexical errors"""
    suggestions = []
    for error in errors:
        error_msg = str(error).lower()
        if 'invalid character' in error_msg:
            suggestions.append("Check for non-ASCII characters or typing errors")
        elif 'unterminated string' in error_msg:
            suggestions.append("Make sure all string literals have closing quotes")
        else:
            suggestions.append("Review the character causing the lexical error")
    return suggestions

def generate_syntax_suggestions(errors):
    """Generate suggestions for syntax errors"""
    suggestions = []
    for error in errors:
        error_msg = str(error).lower()
        if 'expected' in error_msg and ';' in error_msg:
            suggestions.append("Add missing semicolon after statement")
        elif 'expected' in error_msg and ('{' in error_msg or '}' in error_msg):
            suggestions.append("Check matching braces for blocks")
        elif 'expected' in error_msg and ('(' in error_msg or ')' in error_msg):
            suggestions.append("Check parentheses in expressions or function calls")
        else:
            suggestions.append("Review syntax near the error location")
    return suggestions

def count_ast_nodes(ast):
    """Count different types of AST nodes"""
    if not ast:
        return {}
    
    counts = {
        'functions': len(ast.get('functions', [])),
        'total_statements': 0,
        'expressions': 0,
        'declarations': 0
    }
    
    def count_nodes(node):
        if isinstance(node, dict):
            node_type = node.get('type', '')
            if 'Statement' in node_type:
                counts['total_statements'] += 1
            elif 'Expression' in node_type:
                counts['expressions'] += 1
            elif 'Declaration' in node_type:
                counts['declarations'] += 1
            
            for value in node.values():
                if isinstance(value, (list, dict)):
                    count_nodes(value)
        elif isinstance(node, list):
            for item in node:
                count_nodes(item)
    
    count_nodes(ast)
    return counts

def format_minilang_code(code, style='standard'):
    """Enhanced code formatting with different styles"""
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    
    # Style configurations
    styles = {
        'standard': {'indent_size': 4, 'brace_style': 'same_line'},
        'compact': {'indent_size': 2, 'brace_style': 'same_line'},
        'expanded': {'indent_size': 4, 'brace_style': 'new_line'}
    }
    
    config = styles.get(style, styles['standard'])
    indent_size = config['indent_size']
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
        
        # Handle closing braces
        if stripped.startswith('}'):
            indent_level = max(0, indent_level - 1)
        
        # Add indentation
        formatted_line = ' ' * (indent_level * indent_size) + stripped
        formatted_lines.append(formatted_line)
        
        # Handle opening braces
        if stripped.endswith('{'):
            indent_level += 1
    
    return '\n'.join(formatted_lines)

def count_formatting_changes(original, formatted):
    """Count the number of formatting changes made"""
    original_lines = original.split('\n')
    formatted_lines = formatted.split('\n')
    
    changes = 0
    for orig, fmt in zip(original_lines, formatted_lines):
        if orig.strip() == fmt.strip() and orig != fmt:
            changes += 1  # Whitespace change
        elif orig.strip() != fmt.strip():
            changes += 1  # Content change
    
    return changes

def calculate_function_complexity(function):
    """Calculate cyclomatic complexity for a function"""
    complexity = 0
    
    def count_control_structures(node):
        nonlocal complexity
        if not isinstance(node, dict):
            return
        
        node_type = node.get('type', '')
        
        if node_type in ['IfStatement', 'WhileStatement', 'ForStatement']:
            complexity += 1
        
        for key, value in node.items():
            if key != 'type':
                if isinstance(value, list):
                    for item in value:
                        count_control_structures(item)
                elif isinstance(value, dict):
                    count_control_structures(value)
    
    if isinstance(function, dict) and 'body' in function:
        for stmt in function['body']:
            count_control_structures(stmt)
    
    return complexity

# Error handlers
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
    return render_template('index.html'), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'success': False, 'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.before_request
def log_request_info():
    if app.debug:
        print(f"Request: {request.method} {request.path}")

@app.after_request
def after_request(response):
    # CORS and security headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    return response

if __name__ == '__main__':
    # debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 60)
    print("ENHANCED MINILANG++ COMPILER WEB INTERFACE")
    print("=" * 60)
    print("Features:")
    print("  ✓ Complete array support")
    print("  ✓ Full optimization pipeline")
    print("  ✓ Code quality analysis")
    print("  ✓ Performance estimation")
    print("  ✓ Enhanced error reporting")
    print("=" * 60)
    print(f"Server URL: http://{host}:{port}")
    print("API Endpoints:")
    print(f"  POST /compile         - Full compilation pipeline")
    print(f"  POST /api/analyze     - Comprehensive code analysis") 
    print(f"  POST /api/optimize    - Optimization analysis")
    print(f"  POST /api/assembly    - Assembly generation")
    print(f"  POST /api/validate    - Syntax validation")
    print(f"  POST /api/format      - Code formatting")
    print(f"  GET  /examples        - Example programs")
    print(f"  GET  /api/stats       - Compilation statistics")
    print("=" * 60)
    
    app.run(host=host, port=port)