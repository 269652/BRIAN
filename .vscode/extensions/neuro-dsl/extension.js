const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const LANGUAGE_ID = 'neuro';

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
  console.log('NeuroSLM DSL extension activated');

  // Register document formatter
  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(LANGUAGE_ID, {
      provideCompletionItems(document, position) {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);

        // Get word being completed
        const wordMatch = linePrefix.match(/\b([a-zA-Z_][a-zA-Z0-9_]*)$/);
        const word = wordMatch ? wordMatch[1] : '';

        const completions = [];

        // Add keywords
        KEYWORDS.forEach(keyword => {
          if (keyword.startsWith(word)) {
            const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
            item.insertText = keyword;
            completions.push(item);
          }
        });

        // Add built-in functions
        BUILTIN_FUNCTIONS.forEach(fn => {
          if (fn.startsWith(word)) {
            const item = new vscode.CompletionItem(fn, vscode.CompletionItemKind.Function);
            item.insertText = fn + '()';
            item.insertText = new vscode.SnippetString(fn + '($0)');
            completions.push(item);
          }
        });

        // Add built-in variables
        BUILTIN_VARS.forEach(v => {
          if (v.startsWith(word)) {
            const item = new vscode.CompletionItem(v, vscode.CompletionItemKind.Variable);
            item.insertText = v;
            completions.push(item);
          }
        });

        // Extract populations from current file for autocomplete
        const text = document.getText();
        const popMatches = text.matchAll(/(?:export\s+)?population\s+(\w+)/g);
        for (const match of popMatches) {
          const popName = match[1];
          if (popName.startsWith(word)) {
            const item = new vscode.CompletionItem(popName, vscode.CompletionItemKind.Class);
            item.insertText = popName;
            completions.push(item);
          }
        }

        return completions;
      }
    })
  );

  // Register linter command
  context.subscriptions.push(
    vscode.commands.registerCommand('neuro-dsl.lint', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== LANGUAGE_ID) {
        vscode.window.showWarningMessage('Not a .neuro file');
        return;
      }
      lintFile(editor.document);
    })
  );

  // Lint on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        lintFile(doc);
      }
    })
  );

  // Lint on open
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        lintFile(doc);
      }
    })
  );

  // Lint on change (with debounce)
  let changeTimeout;
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (event.document.languageId === LANGUAGE_ID) {
        clearTimeout(changeTimeout);
        changeTimeout = setTimeout(() => {
          lintFile(event.document);
        }, 1000);
      }
    })
  );
};

/**
 * Run the Python linter on a document and update diagnostics
 */
function lintFile(document) {
  const filePath = document.uri.fsPath;
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);

  if (!workspaceFolder) {
    return;
  }

  // Find python executable
  const pythonCmd = findPythonExecutable(workspaceFolder);
  if (!pythonCmd) {
    console.error('Python not found');
    return;
  }

  // Run linter with JSON output
  const proc = spawn(pythonCmd, [
    '-m', 'neuroslm.dsl.neuro_linter',
    filePath,
    '--json'
  ], {
    cwd: workspaceFolder.uri.fsPath,
    encoding: 'utf8'
  });

  let output = '';
  let errorOutput = '';

  proc.stdout.on('data', (data) => {
    output += data.toString();
  });

  proc.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  proc.on('close', (code) => {
    const diags = [];

    try {
      const results = JSON.parse(output);
      if (Array.isArray(results)) {
        results.forEach(result => {
          const range = new vscode.Range(
            new vscode.Position(result.line - 1, result.col - 1),
            new vscode.Position(result.line - 1, result.col + 10)
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
    } catch (e) {
      console.error('Failed to parse linter output:', e);
    }

    diagnosticCollection.set(document.uri, diags);
  });
}

/**
 * Find Python executable in the workspace venv
 */
function findPythonExecutable(workspaceFolder) {
  const basePath = workspaceFolder.uri.fsPath;

  // Try common venv locations
  const candidates = [
    path.join(basePath, '.venv', 'Scripts', 'python.exe'),  // Windows
    path.join(basePath, '.venv', 'bin', 'python'),           // Unix
    path.join(basePath, 'venv', 'Scripts', 'python.exe'),    // Windows
    path.join(basePath, 'venv', 'bin', 'python'),            // Unix
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Fall back to system python
  return process.platform === 'win32' ? 'python.exe' : 'python3';
}

exports.deactivate = function() {
  diagnosticCollection.clear();
  diagnosticCollection.dispose();
};
