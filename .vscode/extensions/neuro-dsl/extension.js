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

/**
 * Search for a reference (@name) in an imported file
 */
function searchImportedFile(currentFileUri, importPath, reference) {
  const baseDir = path.dirname(currentFileUri.fsPath);
  const refName = reference.slice(1); // remove @ prefix

  // Try candidate paths
  const candidates = [
    path.join(baseDir, `${importPath}.neuro`),
    path.join(baseDir, importPath, 'index.neuro'),
    path.join(baseDir, importPath)
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      try {
        const content = fs.readFileSync(candidate, 'utf8');
        const patterns = [
          new RegExp(`\\b(export\\s+)?(equation|dynamics|function|population)\\s+${refName}\\s*[{:]`),
          new RegExp(`\\b(equation|dynamics|function)\\s+${refName}\\s*\\{`)
        ];

        for (const pattern of patterns) {
          const match = pattern.exec(content);
          if (match) {
            const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === candidate);
            if (doc) {
              const matchPos = match.index;
              const lineNum = doc.positionAt(matchPos).line;
              return new vscode.Location(
                vscode.Uri.file(candidate),
                new vscode.Position(lineNum, 0)
              );
            }
          }
        }
      } catch (e) {
        // Silently skip unreadable files
      }
    }
  }

  return null;
}

exports.activate = function(context) {
  OUTPUT_CHANNEL.appendLine('NeuroSLM DSL extension activating...');

  // Register definition provider for @references (Ctrl+Click, F12, Go to Definition)
  context.subscriptions.push(
    vscode.languages.registerDefinitionProvider(LANGUAGE_ID, {
      provideDefinition(document, position) {
        const word = document.getWordRangeAtPosition(position, /@[\w]+/);
        if (!word) return null;

        const reference = document.getText(word); // includes @ prefix
        const text = document.getText();

        // Search for definition: "export equation|dynamics|function|population NAME {" or similar
        const patterns = [
          new RegExp(`\\b(export\\s+)?(equation|dynamics|function|population)\\s+${reference.slice(1)}\\s*[{:]`, 'g'),
          new RegExp(`\\b(equation|dynamics|function)\\s+${reference.slice(1)}\\s*\\{`, 'g')
        ];

        for (const pattern of patterns) {
          const match = pattern.exec(text);
          if (match) {
            const matchPos = match.index;
            const lineNum = document.positionAt(matchPos).line;
            const charNum = document.positionAt(matchPos).character;
            return new vscode.Location(
              document.uri,
              new vscode.Position(lineNum, charNum)
            );
          }
        }

        // If not found in current file, search in imported files
        const importMatches = [...text.matchAll(/import\s*\{[^}]*\}\s*from\s*["'](@?[^"']+)["']/g)];
        for (const importMatch of importMatches) {
          const importPath = importMatch[1].replace(/^@\//, '');
          const importedDef = searchImportedFile(document.uri, importPath, reference);
          if (importedDef) return importedDef;
        }

        return null;
      }
    })
  );

  // Register code action provider for autofix
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(LANGUAGE_ID, {
      provideCodeActions(document, range, context) {
        const actions = [];

        // Check diagnostics in this range
        for (const diagnostic of context.diagnostics) {
          if (diagnostic.code === 'repeated-equation') {
            const action = new vscode.CodeAction(
              'Extract as equation definition',
              vscode.CodeActionKind.QuickFix
            );
            action.command = {
              title: 'Extract as equation definition',
              command: 'neuro-dsl.extractEquation',
              arguments: [document, diagnostic]
            };
            action.diagnostics = [diagnostic];
            actions.push(action);
          }
        }

        return actions;
      }
    })
  );

  // Register command for extracting equations
  context.subscriptions.push(
    vscode.commands.registerCommand('neuro-dsl.extractEquation', async (document, diagnostic) => {
      // Get equation name from user
      const equationName = await vscode.window.showInputBox({
        prompt: 'Enter equation name (e.g., standard_synapse)',
        placeHolder: 'equation_name',
        validateInput: (value) => {
          if (!value.match(/^[a-z_][a-z0-9_]*$/i)) {
            return 'Name must be a valid identifier (letters, numbers, underscores)';
          }
          return null;
        }
      });

      if (!equationName) return;

      const text = document.getText();
      const lines = text.split('\n');

      // Extract the equation from the diagnostic message
      const match = text.match(/formula: "([^"]*)"/);
      if (!match) {
        vscode.window.showErrorMessage('Could not extract equation formula');
        return;
      }

      const formula = match[1];

      // Find all parameters in the formula
      const params = [];
      const paramMatch = formula.match(/\b([a-zA-Z_]\w*)\b/g);
      if (paramMatch) {
        const builtins = new Set(['ReLU', 'sigmoid', 'tanh', 'sin', 'cos', 'exp', 'log', 'sqrt', 'matmul', 'x', 'y', 's', 'V', 'dt', 'pi', 'e']);
        paramMatch.forEach(p => {
          if (!builtins.has(p) && !params.includes(p)) {
            params.push(p);
          }
        });
      }

      // Create equation definition
      const equationDef = `export equation ${equationName} {\n    params: [${params.join(', ')}],\n    formula: "${formula}"\n}\n`;

      // Insert equation definition at top of file
      const insertPos = new vscode.Position(0, 0);
      const edit = new vscode.WorkspaceEdit();
      edit.insert(document.uri, insertPos, equationDef + '\n');

      // Replace all repeated equations with reference
      const fullText = document.getText();
      const replacementEdit = fullText.replace(new RegExp(`equation: "${formula.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g'), `equation: @${equationName}`);

      edit.replace(
        document.uri,
        new vscode.Range(new vscode.Position(0, 0), new vscode.Position(lines.length, 0)),
        equationDef + '\n' + replacementEdit
      );

      await vscode.workspace.applyEdit(edit);
      vscode.window.showInformationMessage(`✓ Extracted equation '${equationName}' and replaced ${(replacementEdit.match(new RegExp(`@${equationName}`, 'g')) || []).length} occurrences`);
    })
  );

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
    OUTPUT_CHANNEL.appendLine(`✗ Linter process error: ${err.message}`);
    OUTPUT_CHANNEL.appendLine('This usually means Python executable not found');
    diagnosticCollection.set(document.uri, []);
  });

  proc.on('close', (code) => {
    const diags = [];

    if (code === 0 || code === 1) {
      // code 0 = success, code 1 = had errors/warnings (both are ok)
      try {
        if (output.trim()) {
          const results = JSON.parse(output);
          if (Array.isArray(results)) {
            OUTPUT_CHANNEL.appendLine(`✓ Linter found ${results.length} diagnostic(s)`);
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
        } else {
          OUTPUT_CHANNEL.appendLine('✓ Linter passed (no issues)');
        }
      } catch (e) {
        OUTPUT_CHANNEL.appendLine(`✗ JSON parse error: ${e.message}`);
        OUTPUT_CHANNEL.appendLine(`Output was: ${output.substring(0, 500)}`);
      }
    } else if (code === null) {
      OUTPUT_CHANNEL.appendLine(`✗ Linter killed or crashed`);
    } else {
      OUTPUT_CHANNEL.appendLine(`✗ Linter failed with code ${code}`);
      OUTPUT_CHANNEL.appendLine(`stderr: ${errorOutput.substring(0, 500)}`);
    }

    diagnosticCollection.set(document.uri, diags);
  });
}

/**
 * Find Python executable in the workspace venv
 */
function findPythonExecutable(workspaceFolder) {
  const basePath = workspaceFolder.uri.fsPath;
  OUTPUT_CHANNEL.appendLine(`Workspace path: ${basePath}`);

  // Try venv locations in order
  const candidates = [
    path.join(basePath, '.venv', 'Scripts', 'python.exe'),
    path.join(basePath, '.venv', 'bin', 'python'),
    path.join(basePath, 'venv', 'Scripts', 'python.exe'),
    path.join(basePath, 'venv', 'bin', 'python'),
  ];

  for (const candidate of candidates) {
    OUTPUT_CHANNEL.appendLine(`Checking: ${candidate}`);
    if (fs.existsSync(candidate)) {
      OUTPUT_CHANNEL.appendLine(`✓ Found Python at: ${candidate}`);
      return candidate;
    }
  }

  OUTPUT_CHANNEL.appendLine('⚠ Venv Python not found, falling back to system Python');
  OUTPUT_CHANNEL.appendLine('This will fail if system Python < 3.7');
  return process.platform === 'win32' ? 'python.exe' : 'python3';
}

exports.deactivate = function() {
  diagnosticCollection.clear();
  diagnosticCollection.dispose();
  OUTPUT_CHANNEL.dispose();
};
