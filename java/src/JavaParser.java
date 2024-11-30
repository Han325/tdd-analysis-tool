// JavaParser.java
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.ImportDeclaration;
import py4j.GatewayServer;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;
import java.util.Set;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicBoolean;

public class JavaParser {
    private static final Logger LOGGER = Logger.getLogger(JavaParser.class.getName());
    
    private static final Set<String> TEST_IMPORTS = new HashSet<String>() {{
        add("org.junit");
        add("org.testng");
        add("org.mockito");
        add("org.easymock");
        add("org.powermock");
        add("junit");
        add("mock");
        add("Assert");
    }};

    private static final Set<String> TEST_ANNOTATIONS = new HashSet<String>() {{
        add("Test");
        add("Before");
        add("After");
        add("BeforeClass");
        add("AfterClass");
    }};
    
    public JavaParser() {
        LOGGER.setLevel(Level.ALL);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.ALL);
        LOGGER.addHandler(handler);
    }
    
    public ParseResult parseJavaContent(String content) {
        try {
            LOGGER.info("Starting to parse Java content");
            
            if (content == null || content.trim().isEmpty()) {
                LOGGER.warning("Received empty or null content");
                return new ParseResult();
            }
            
            CompilationUnit cu = StaticJavaParser.parse(content);
            ParseResult result = new ParseResult();
            
            // Use AtomicBoolean for thread-safe mutable flags
            AtomicBoolean hasTestImports = new AtomicBoolean(false);
            AtomicBoolean hasTestAnnotations = new AtomicBoolean(false);
            
            // Get imports and check for test frameworks
            for (ImportDeclaration imp : cu.getImports()) {
                String importName = imp.getNameAsString();
                result.addImport(importName);
                
                // Check if this is a test-related import
                if (TEST_IMPORTS.stream().anyMatch(importName::contains)) {
                    hasTestImports.set(true);
                    LOGGER.fine("Found test import: " + importName);
                }
            }
            
            // Process classes
            cu.findAll(ClassOrInterfaceDeclaration.class).forEach(cls -> {
                String className = cls.getNameAsString();
                result.addClass(className);
                result.setAbstract(cls.isAbstract());
                
                // Check for test annotations on class
                if (cls.getAnnotations().stream()
                        .anyMatch(ann -> TEST_ANNOTATIONS.contains(ann.getNameAsString()))) {
                    hasTestAnnotations.set(true);
                }
                
                // Check if utility class (all static methods)
                boolean allStatic = cls.getMethods().stream()
                    .allMatch(method -> method.isStatic());
                result.setUtility(allStatic);
                
                // Process methods
                cls.getMethods().forEach(method -> {
                    String methodName = method.getNameAsString();
                    result.addMethod(methodName);
                    
                    // Check for test annotations on methods
                    if (method.getAnnotations().stream()
                            .anyMatch(ann -> TEST_ANNOTATIONS.contains(ann.getNameAsString()))) {
                        hasTestAnnotations.set(true);
                        result.addTestMethod(methodName);
                    }
                    // Check for test method naming patterns
                    else if (methodName.startsWith("test") || 
                             methodName.contains("should") || 
                             methodName.contains("verify")) {
                        result.addTestMethod(methodName);
                    }
                });
            });
            
            // Set test-related flags
            result.setHasTestImports(hasTestImports.get());
            result.setHasTestAnnotations(hasTestAnnotations.get());
            
            LOGGER.info("Successfully created ParseResult object");
            return result;
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error parsing Java content", e);
            return new ParseResult();
        }
    }

    public static void main(String[] args) {
        try {
            JavaParser parser = new JavaParser();
            GatewayServer server = new GatewayServer(parser);
            server.start();
            LOGGER.info("Gateway Server Started on default port (25333)");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Failed to start Gateway Server", e);
        }
    }
}