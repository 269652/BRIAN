const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const LANGUAGE_ID = 'neuro';
const OUTPUT_CHANNEL = vscode.window.createOutputChannel('NeuroSLM DSL');

// Collect diagnostics from linter output
const diagnosticCollection = vscode.languages.createDiagnosticCollection('neuro');

// Linting keywords for autocomplete
const KEYWORDS = [
  'architecture', 'neurotransmitter', 'population', 'export', 'synapse',
  'import', 'from', 'training', 'dynamics', 'function', 'mechanisms',
  'count', 'equation', 'ode', 'weight', 'dynamics', 'timescale', 'capacity'
];

const BUILTIN_FUNCTIONS = [
  'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'sigmoid',
  'ReLU', 'silu', 'gelu', 'swiglu', 'matmul', 'linear', 'rmsnorm',
  'causal_self_attention', 'embedding', 'softmax', 'dropout', 'layer_norm'
];

const BUILTIN_VARS = [
  'x', 'y', 's', 'V', 'd_sem', 'dt', 'tau', 'weight',
  'x_pre', 'x_post', 'e', 'pi', 'nan', 'inf'
];

exports.activate = function(context) {
  OUTPUT_CHANNEL.appendLine('NeuroSLM DSL extension activating...');

  // Register autocomplete provider
  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(LANGUAGE_ID, {
      provideCompletionItems(document, position) {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        const wordMatch = linePrefix.match(/\b([a-zA-Z_][a-zA-Z0-9_]*)$/);
        const word = wordMatch ? wordMatch[1] : '';

        const completions = [];

        // Add keywords
        KEYWORDS.forEach(keyword => {
          if (keyword.startsWith(word)) {
            const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
            completions.push(item);
          }
        });

        // Add built-in functions with snippet
        BUILTIN_FUNCTIONS.forEach(fn => {
          if (fn.startsWith(word)) {
            const item = new vscode.CompletionItem(fn, vscode.CompletionItemKind.Function);
            item.insertText = new vscode.SnippetString(fn + '($0)');
            completions.push(item);
          }
        });

        // Add built-in variables
        BUILTIN_VARS.forEach(v => {
          if (v.startsWith(word)) {
            const item = new vscode.CompletionItem(v, vscode.CompletionItemKind.Variable);
            completions.push(item);
          }
        });

        // Extract populations from current file
        const text = document.getText();
        const popMatches = text.matchAll(/(?:export\s+)?population\s+(\w+)/g);
        for (const match of popMatches) {
          const popName = match[1];
          if (popName.startsWith(word)) {
            const item = new vscode.CompletionItem(popName, vscode.CompletionItemKind.Class);
            completions.push(item);
          }
        }

        return completions;
      }
    })
  );

  // Lint on open
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        OUTPUT_CHANNEL.appendLine(`Linting on open: ${doc.fileName}`);
        lintFile(doc);
      }
    })
  );

  // Lint on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        OUTPUT_CHANNEL.appendLine(`Linting on save: ${doc.fileName}`);
        lintFile(doc);
      }
    })
  );

  // Lint on change (debounced)
  let changeTimeout;
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (event.document.languageId === LANGUAGE_ID) {
        clearTimeout(changeTimeout);
        changeTimeout = setTimeout(() => {
          lintFile(event.document);
        }, 1500);
      }
    })
  );

  OUTPUT_CHANNEL.appendLine('✓ NeuroSLM DSL extension activated successfully');
};

/**
 * Run the Python linter on a document
 */
function lintFile(document) {
  const filePath = document.uri.fsPath;
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);

  if (!workspaceFolder) {
    OUTPUT_CHANNEL.appendLine('No workspace folder found');
    return;
  }

  const pythonCmd = findPythonExecutable(workspaceFolder);
  OUTPUT_CHANNEL.appendLine(`Using Python: ${pythonCmd}`);

  if (!pythonCmd) {
    OUTPUT_CHANNEL.appendLine('Error: Python executable not found');
    return;
  }

  // Run linter
  const proc = spawn(pythonCmd, [
    '-m', 'neuroslm.dsl.neuro_linter',
    filePath,
    '--json'
  ], {
    cwd: workspaceFolder.uri.fsPath
  });

  let output = '';
  let errorOutput = '';

  proc.stdout.on('data', (data) => {
    output += data.toString();
  });

  proc.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  proc.on('error', (err) => {
    OUTPUT_CHANNEL.appendLine(`Linter error: ${err.message}`);
  });

  proc.on('close', (code) => {
    const diags = [];

    if (code === 0 || code === 1) {
      // code 0 = success, code 1 = had errors/warnings (both are ok)
      try {
        if (output.trim()) {
          const results = JSON.parse(output);
          if (Array.isArray(results)) {
            OUTPUT_CHANNEL.appendLine(`Found ${results.length} diagnostic(s)`);
            results.forEach(result => {
              const range = new vscode.Range(
                new vscode.Position(result.line - 1, Math.max(0, result.col - 1)),
                new vscode.Position(result.line - 1, Math.min(result.col + 10, 999))
              );

              const severity = {
                'error': vscode.DiagnosticSeverity.Error,
                'warning': vscode.DiagnosticSeverity.Warning,
                'info': vscode.DiagnosticSeverity.Information
              }[result.severity] || vscode.DiagnosticSeverity.Information;

              const diag = new vscode.Diagnostic(
                range,
                `[${result.code}] ${result.message}`,
                severity
              );
              diag.source = 'neuro-linter';
              diags.push(diag);
            });
          }
        }
      } catch (e) {
        OUTPUT_CHANNEL.appendLine(`Parse error: ${e.message}`);
        OUTPUT_CHANNEL.appendLine(`Output was: ${output}`);
      }
    } else {
      OUTPUT_CHANNEL.appendLine(`Linter failed with code ${code}: ${errorOutput}`);
    }

    diagnosticCollection.set(document.uri, diags);
  });
}

/**
 * Find Python executable in the workspace venv
 */
function findPythonExecutable(workspaceFolder) {
  const basePath = workspaceFolder.uri.fsPath;

  // Try venv locations in order
  const candidates = [
    path.join(basePath, '.venv', 'Scripts', 'python.exe'),
    path.join(basePath, '.venv', 'bin', 'python'),
    path.join(basePath, 'venv', 'Scripts', 'python.exe'),
    path.join(basePath, 'venv', 'bin', 'python'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      OUTPUT_CHANNEL.appendLine(`Found Python at: ${candidate}`);
      return candidate;
    }
  }

  OUTPUT_CHANNEL.appendLine('Venv Python not found, using system Python');
  return process.platform === 'win32' ? 'python.exe' : 'python3';
}

exports.deactivate = function() {
  diagnosticCollection.clear();
  diagnosticCollection.dispose();
  OUTPUT_CHANNEL.dispose();
};
